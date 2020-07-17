
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from functools import reduce
from itertools import chain, zip_longest

import cv2
import numpy as np

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.config import Config, SchemaBuilder
from datumaro.components.extractor import AnnotationType, Bbox, LabelCategories
from datumaro.components.project import Dataset
from datumaro.util import find
from datumaro.util.annotation_tools import compute_bbox, iou as segment_iou, nms

SEGMENT_TYPES = {
    AnnotationType.bbox,
    AnnotationType.polygon,
    AnnotationType.mask
}

def get_segments(anns, conf_threshold=0.0):
    return [ann for ann in anns \
        if conf_threshold <= ann.attributes.get('score', 1) and \
            ann.type in SEGMENT_TYPES
    ]

def merge_annotations_unique(a, b):
    merged = []
    for item in chain(a, b):
        found = False
        for elem in merged:
            if elem == item:
                found = True
                break
        if not found:
            merged.append(item)

    return merged

def merge_categories(sources):
    categories = {}
    for source in sources:
        categories.update(source)
    for source in sources:
        for cat_type, source_cat in source.items():
            if not categories[cat_type] == source_cat:
                raise NotImplementedError(
                    "Merging different categories is not implemented yet")
    return categories


class MergingStrategy(CliPlugin):
    @classmethod
    def merge(cls, sources, **options):
        instance = cls(**options)
        return instance(sources)

    def __init__(self, **options):
        self._conf = Config(options,
            fallback=getattr(self, 'CONFIG_DEFAULTS', None),
            schema=getattr(self, 'CONFIG_SCHEMA', None),
            mutable=False)
        print(self._conf)
        self._sources = None

    def __call__(self, sources):
        raise NotImplementedError()

    def __getattr__(self, name):
        return self.__dict__.get('_conf', {}).get(name)

class IntersectMerge(MergingStrategy):
    # TODO: put this class to the right place
    CONFIG_SCHEMA = SchemaBuilder() \
        .add('iou_thresh', float) \
        .add('do_nms', bool) \
        .add('input_conf_thresh', float) \
        .add('output_conf_thresh', float) \
        .add('quorum', int) \
        .add('ignored_attributes', set) \
        .build()

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        pass

    _image = None # TODO: remove after debug
    _item = None # TODO: remove after debug
    def __call__(self, datasets):
        merged = Dataset(
            categories=merge_categories(d.categories() for d in datasets))

        item_ids = set((item.id, item.subset) for d in datasets for item in d)

        for (item_id, item_subset) in item_ids:
            items = []
            for i, d in enumerate(datasets):
                try:
                    items.append(d.get(item_id, subset=item_subset))
                except KeyError:
                    log.debug("Source #%s doesn't have item '%s' in subset '%s'",
                        1 + i + 1, item_id, item_subset)
            merged.put(self.merge_items(items))
        return merged

    def merge_items(self, items):
        self._item = items[0]

        sources = [[a for a in item.annotations
            if self.input_conf_thresh <= a.attributes.get('score', 1)
            ] for item in items]
        if self.do_nms:
            sources = list(map(nms, sources))
        print(list(map(len, sources)))

        annotations = self.merge_annotations(sources)

        annotations = [a for a in annotations
            if self.output_conf_thresh <= a.attributes.get('score', 1)]

        return items[0].wrap(annotations=annotations)

    def merge_annotations(self, sources):
        all_by_type = {}
        for s in sources:
            src_by_type = {}
            for a in s:
                src_by_type.setdefault(a.type, []).append(a)
            for k, v in src_by_type.items():
                all_by_type.setdefault(a.type, []).append(v)

        annotations = []
        for k, v in all_by_type.items():
            annotations += self._merge_annotations_by_type(k, v)

        return annotations

    def _merge_annotations_by_type(self, t, ann):
        if t is AnnotationType.label:
            return LabelMergingStrategy(self.quorum).merge(ann)
        elif t is AnnotationType.bbox:
            return BboxMergingStrategy(self.cluster_iou, self.pairwise_iou,
                self.quorum, self.ignored_attributes).merge(ann)
        elif t is AnnotationType.mask:
            return MaskMergingStrategy(self.cluster_iou, self.pairwise_iou,
                self.quorum, self.ignored_attributes).merge(ann)
        elif t is AnnotationType.polygon:
            return PolygonMergingStrategy(self.cluster_iou, self.pairwise_iou,
                self.quorum, self.ignored_attributes).merge(ann)
        elif t is AnnotationType.polyline:
            return LineMergingStrategy(self.cluster_iou, self.pairwise_iou,
                self.quorum, self.ignored_attributes).merge(ann)
        elif t is AnnotationType.points:
            return PointsMergingStrategy(self.cluster_iou, self.pairwise_iou,
                self.quorum, self.ignored_attributes).merge(ann)
        elif t is AnnotationType.caption:
            return reduce(merge_annotations_unique, ann, [])
        else:
            raise NotImplementedError("Merge for %s type is not supported" % t)

class LabelMergingStrategy:
    def __init__(self, quorum=0):
        self._quorum = quorum

    def merge(self, sources):
        votes = {} # label -> score
        for s in chain(*sources):
            for label_ann in s:
                votes[label_ann.label] = 1.0 + votes.get(value, 0.0)

        labels = {}
        for name, votes in votes.items():
            votes_count, label = max(votes.items(), key=lambda e: e[1])
            if self.quorum <= votes_count:
                labels[name] = label

        return labels

class ShapeMergingStrategy:
    def __init__(self, cluster_dist=None, pairwise_dist=None, quorum=0,
            ignored_attributes=None):
        self.cluster_dist = cluster_dist
        self.pairwise_dist = pairwise_dist
        self.quorum = quorum
        self.ignored_attributes = ignored_attributes or set()

    def merge(self, sources):
        clusters, _ = self.find_segment_clusters(sources)
        group_map = self.find_cluster_groups(clusters)

        merged = []
        for cluster_id, cluster in enumerate(clusters):
            label, label_score = self.find_cluster_label(cluster)
            shape, shape_score = self.merge_cluster_shape(cluster)

            attributes = self.find_cluster_attrs(cluster)
            attributes = { k: v for k, v in attributes.items()
                if k not in self.ignored_attributes }

            attributes['score'] = label_score * shape_score \
                if label is not None else shape_score

            group_id, (cluster_group, ann_groups) = find(enumerate(group_map),
                lambda e: cluster_id in e[1][0])
            if not ann_groups:
                group_id = None

            shape.label = label
            shape.group = group_id
            shape.attributes.update(attributes)

            cv2.rectangle(self._image, (int(bbox[0]), int(bbox[1])),
                (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                (255, 255, 255), thickness=1)

            merged.append(shape)

        from datumaro.util.image import save_image
        save_image('test_clusters/%s.jpg' % self._item.id, self._image,
            create_dir=True)

        return merged

    @staticmethod
    def find_cluster_label(cluster):
        quorum = self.quorum or 0

        label_votes = {}
        votes_count = 0
        for s in cluster:
            if s.label is None:
                continue

            weight = s.attributes.get('score', 1.0)
            label_votes[s.label] = weight + label_votes.get(s.label, 0.0)
            votes_count += 1

        if votes_count < quorum:
            return None, None

        label, score = max(label_votes.items(), key=lambda e: e[1], default=None)
        score = score / votes_count if votes_count else None
        return label, score

    @staticmethod
    def find_cluster_groups(clusters):
        cluster_groups = []
        visited = set()
        for a_idx, cluster_a in enumerate(clusters):
            if a_idx in visited:
                continue
            visited.add(a_idx)

            cluster_group = { a_idx }

            # find segment groups in the cluster group
            a_groups = set(ann.group for ann in cluster_a)
            for cluster_b in clusters[a_idx+1 :]:
                b_groups = set(ann.group for ann in cluster_b)
                if a_groups & b_groups:
                    a_groups |= b_groups

            # now we know all the segment groups in this cluster group
            # so we can find adjacent clusters
            for b_idx, cluster_b in enumerate(clusters[a_idx+1 :]):
                b_idx = a_idx + 1 + b_idx
                b_groups = set(ann.group for ann in cluster_b)
                if a_groups & b_groups:
                    cluster_group.add(b_idx)
                    visited.add(b_idx)

            cluster_groups.append( (cluster_group, a_groups) )
        return cluster_groups

    def find_cluster_attrs(self, cluster):
        quorum = self.quorum or 0

        # TODO: when attribute types are implemented, add linear
        # interpolation for contiguous values

        attr_votes = {} # name -> { value: score , ... }
        for s in cluster:
            for name, value in s.attributes.items():
                votes = attr_votes.get(name, {})
                votes[value] = 1.0 + votes.get(value, 0.0)
                attr_votes[name] = votes

        attributes = {}
        for name, votes in attr_votes.items():
            vote, count = max(votes.items(), key=lambda e: e[1])
            if count < quorum:
                continue
            attributes[name] = vote

        return attributes

    def find_segment_clusters(self, sources, types=None):
        if types is None:
            types = SEGMENT_TYPES
        sources = [[a for a in source if a.type in types] for source in sources]
        clusters, id_segm = self.find_clusters(sources)

        return clusters, id_segm

    def find_clusters(self, sources):
        distance = self.distance
        pairwise_dist = self.pairwise_dist
        cluster_dist = self.cluster_dist

        if pairwise_dist is None: pairwise_dist = 0.9
        if cluster_dist is None: cluster_dist = pairwise_dist

        id_segm = { id(sgm): (sgm, src_i)
            for src_i, src in enumerate(sources) for sgm in src }

        def _is_close_enough(cluster, extra_id):
            # check if whole cluster IoU will not be broken
            # when this segment is added
            b = id_segm[extra_id][0]
            for a_id in cluster:
                a = id_segm[a_id][0]
                if distance(a, b) < cluster_dist:
                    return False
            return True

        def _has_same_source(cluster, extra_id):
            b = id_segm[extra_id][1]
            for a_id in cluster:
                a = id_segm[a_id][1]
                if a == b:
                    return True
            return False

        # match segments in sources, pairwise
        adjacent = { i: [] for i in id_segm } # id(sgm) -> [id(adj_sgm1), ...]
        for a_idx, src_a in enumerate(sources):
            for src_b in sources[a_idx+1 :]:
                matches, _, _, _ = compare_segments(src_a, src_b,
                    dist_thresh=pairwise_dist, distance=distance)
                for m in matches:
                    adjacent[id(m[0])].append(id(m[1]))

        # join all segments into matching clusters
        clusters = []
        visited = set()
        for cluster_idx in adjacent:
            if cluster_idx in visited:
                continue

            cluster = set()
            to_visit = { cluster_idx }
            while to_visit:
                c = to_visit.pop()
                cluster.add(c)
                visited.add(c)

                for i in adjacent[c]:
                    if i in visited:
                        continue
                    if 0 < cluster_dist and not _is_close_enough(cluster, i):
                        continue
                    if _has_same_source(cluster, i):
                        continue

                    to_visit.add(i)

            clusters.append([id_segm[i][0] for i in cluster])

        return clusters, id_segm

    @staticmethod
    def distance(a, b):
        return segment_iou(a, b)

    def _merge_cluster_shape_mean_box_nearest(self, cluster):
        le = len(cluster)
        boxes = [s.get_bbox() for s in cluster]
        mlb = sum(b[0] for b in boxes) / le
        mrb = sum(b[1] for b in boxes) / le
        mtb = sum(b[0] + b[2] for b in boxes) / le
        mbb = sum(b[1] + b[3] for b in boxes) / le
        mbbox = [mlb, mtb, mrb - mlb, mbb - mtb]

        dist = (self.distance(mbbox, s) for s in cluster)
        min_dist_pos, _ = min(enumerate(dist), key=lambda e: e[1])
        return cluster[min_dist_pos]

    def merge_cluster_shape(self, cluster):
        shape = self._merge_cluster_shape_mean_box_nearest(cluster)
        shape_score = sum(max(0, self.distance(shape, s))
            for s in cluster) / len(cluster)
        return shape, shape_score

class BboxMergingStrategy(ShapeMergingStrategy):
    def find_segment_clusters(self, sources, types=None):
        clusters, id_segm = super().find_segment_clusters(sources,
            AnnotationType.bbox)

        # debugging code
        if distance == segment_iou:
            import datumaro.util.mask_tools as mask_tools
            self._image = self._item.image.data.copy()
            cm = mask_tools.generate_colormap()
            for c_id, cluster in enumerate(clusters):
                color = tuple(map(int, cm[1 + c_id][::-1]))
                for segm in cluster:
                    _, src_id = id_segm[id(segm)]
                    cv2.rectangle(self._image, (int(segm.x), int(segm.y)),
                        ((int(segm.points[2]), int(segm.points[3]))), color,
                        thickness=2)
                    cv2.putText(self._image, '(c%s,s%s)' % (c_id, src_id),
                        (int(segm.x) + 1, int(segm.y) + 1),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
                    cv2.putText(self._image, '(c%s,s%s)' % (c_id, src_id),
                        (int(segm.x), int(segm.y)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color)

        return clusters, id_segm

class PolygonMergingStrategy(ShapeMergingStrategy):
    def find_segment_clusters(self, sources, types=None):
        return super().find_segment_clusters(sources, AnnotationType.polygon)

class MaskMergingStrategy(ShapeMergingStrategy):
    def find_segment_clusters(self, sources, types=None):
        return super().find_segment_clusters(sources, AnnotationType.mask)

class PointsMergingStrategy(ShapeMergingStrategy):
    @staticmethod
    def distance(a, b):
        def _descriptor(points):
            # use mean filter to prevent outliers from big impact
            cm = np.mean(points, axis=0)
            weights = np.linalg.norm(points - cm, axis=1)
            weights /= np.sum(weights)

            return points * weights

        d1 = _descriptor(a.points.reshape((-1, 2)))
        d2 = _descriptor(b.points.reshape((-1, 2)))
        d = np.linalg.norm(d1 - d2, axis=1)

        return 1 / np.cosh(np.sum(d))

    def find_segment_clusters(self, sources, types=None):
        return super().find_segment_clusters(sources, AnnotationType.points)

class LineMergingStrategy:
    @staticmethod
    def distance(line1, line2):
        point_count = max(len(line1.points) // 2, len(line2.points) // 2)
        line1 = smooth_line(line1.points, point_count)
        line2 = smooth_line(line2.points, point_count)

        p1 = np.linalg.norm(line1, axis=1)
        p1 /= np.linalg.norm(p1)

        p2 = np.linalg.norm(line2, axis=1)
        p2 /= np.linalg.norm(p2)

        return abs(np.dot(p1, p2))

    def find_segment_clusters(self, sources, types=None):
        return super().find_segment_clusters(sources, AnnotationType.polyline)

def smooth_line(points, segments):
    assert 2 <= len(points) // 2 and len(points) % 2 == 0

    if len(points) // 2 == segments:
        return points

    points = list(points)
    if len(points) == 2:
        points.extend(points)
    points = np.array(points).reshape((-1, 2))

    lengths = np.sqrt(np.square(points[1:] - points[:-1]))
    dists = [0]
    for l in lengths:
        dists.append(dists[-1] + l)

    length = np.sum(lengths)
    step = length / segments

    new_points = np.zeros((segments + 1, 2))
    new_points[0] = points[0]

    last_segment = 0
    for segment_idx in range(segments):
        pos = segment_idx * step

        while dists[last_segment + 1] < pos:
            last_segment += 1

        segment_start = dists[last_segment]
        segment_len = lengths[segment_idx]
        prev_p = points[last_segment]
        next_p = points[last_segment + 1]
        r = (pos - segment_start) / segment_len

        new_points[segment_idx + 1] = prev_p * (1 - r) + next_p * r

    return new_points

def compare_segments(a_segms, b_segms, distance='iou', dist_thresh=1.0):
    if distance == 'iou':
        distance = segment_iou
    else:
        assert callable(distance)

    a_segms.sort(key=lambda ann: 1 - ann.attributes.get('score', 1))
    b_segms.sort(key=lambda ann: 1 - ann.attributes.get('score', 1))

    # a_matches: indices of b_segms matched to a bboxes
    # b_matches: indices of a_segms matched to b bboxes
    a_matches = -np.ones(len(a_segms), dtype=int)
    b_matches = -np.ones(len(b_segms), dtype=int)

    distances = np.array([[distance(a, b) for b in b_segms] for a in a_segms])

    # matches: boxes we succeeded to match completely
    # mispred: boxes we succeeded to match, having label mismatch
    matches = []
    mispred = []

    for a_idx, a_segm in enumerate(a_segms):
        if len(b_segms) == 0:
            break
        matched_b = a_matches[a_idx]
        max_dist = max(distances[a_idx, matched_b], dist_thresh)
        for b_idx, b_segm in enumerate(b_segms):
            if 0 <= b_matches[b_idx]: # assign a_segm with max conf
                continue
            d = distances[a_idx, b_idx]
            if d < max_dist:
                continue
            max_dist = d
            matched_b = b_idx

        if matched_b < 0:
            continue
        a_matches[a_idx] = matched_b
        b_matches[matched_b] = a_idx

        b_segm = b_segms[matched_b]

        if a_segm.label == b_segm.label:
            matches.append( (a_segm, b_segm) )
        else:
            mispred.append( (a_segm, b_segm) )

    # *_umatched: boxes of (*) we failed to match
    a_unmatched = [a_segms[i] for i, m in enumerate(a_matches) if m < 0]
    b_unmatched = [b_segms[i] for i, m in enumerate(b_matches) if m < 0]

    return matches, mispred, a_unmatched, b_unmatched

class Comparator:
    def __init__(self, iou_threshold=0.5, conf_threshold=0.9):
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    # pylint: disable=no-self-use
    def compare_dataset_labels(self, extractor_a, extractor_b):
        a_label_cat = extractor_a.categories().get(AnnotationType.label)
        b_label_cat = extractor_b.categories().get(AnnotationType.label)
        if not a_label_cat and not b_label_cat:
            return None
        if not a_label_cat:
            a_label_cat = LabelCategories()
        if not b_label_cat:
            b_label_cat = LabelCategories()

        mismatches = []
        for a_label, b_label in zip_longest(a_label_cat.items, b_label_cat.items):
            if a_label != b_label:
                mismatches.append((a_label, b_label))
        return mismatches
    # pylint: enable=no-self-use

    def compare_item_labels(self, item_a, item_b):
        conf_threshold = self.conf_threshold

        a_labels = set(ann.label for ann in item_a.annotations \
            if ann.type is AnnotationType.label and \
               conf_threshold < ann.attributes.get('score', 1))
        b_labels = set(ann.label for ann in item_b.annotations \
            if ann.type is AnnotationType.label and \
               conf_threshold < ann.attributes.get('score', 1))

        a_unmatched = a_labels - b_labels
        b_unmatched = b_labels - a_labels
        matches = a_labels & b_labels

        return matches, a_unmatched, b_unmatched

    def compare_item_bboxes(self, item_a, item_b):
        a_boxes = get_segments(item_a.annotations, self.conf_threshold)
        b_boxes = get_segments(item_b.annotations, self.conf_threshold)
        return compare_segments(a_boxes, b_boxes,
            dist_thresh=self.iou_threshold)

def mean_std(dataset):
    """
    Computes unbiased mean and std. dev. for dataset images, channel-wise.
    """
    # Use an online algorithm to:
    # - handle different image sizes
    # - avoid cancellation problem

    stats = np.empty((len(dataset), 2, 3), dtype=np.double)
    counts = np.empty(len(dataset), dtype=np.uint32)

    mean = lambda i, s: s[i][0]
    var = lambda i, s: s[i][1]

    for i, item in enumerate(dataset):
        counts[i] = np.prod(item.image.size)

        image = item.image.data
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        else:
            image = image[:, :, :3]
        # opencv is much faster than numpy here
        cv2.meanStdDev(image.astype(np.double) / 255,
            mean=mean(i, stats), stddev=var(i, stats))

    # make variance unbiased
    np.multiply(np.square(stats[:, 1]),
        (counts / (counts - 1))[:, np.newaxis],
        out=stats[:, 1])

    _, mean, var = StatsCounter().compute_stats(stats, counts, mean, var)
    return mean * 255, np.sqrt(var) * 255

class StatsCounter:
    # Implements online parallel computation of sample variance
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    # Needed do avoid catastrophic cancellation in floating point computations
    @staticmethod
    def pairwise_stats(count_a, mean_a, var_a, count_b, mean_b, var_b):
        delta = mean_b - mean_a
        m_a = var_a * (count_a - 1)
        m_b = var_b * (count_b - 1)
        M2 = m_a + m_b + delta ** 2 * count_a * count_b / (count_a + count_b)
        return (
            count_a + count_b,
            mean_a * 0.5 + mean_b * 0.5,
            M2 / (count_a + count_b - 1)
        )

    # stats = float array of shape N, 2 * d, d = dimensions of values
    # count = integer array of shape N
    # mean_accessor = function(idx, stats) to retrieve element mean
    # variance_accessor = function(idx, stats) to retrieve element variance
    # Recursively computes total count, mean and variance, does O(log(N)) calls
    @staticmethod
    def compute_stats(stats, counts, mean_accessor, variance_accessor):
        m = mean_accessor
        v = variance_accessor
        n = len(stats)
        if n == 1:
            return counts[0], m(0, stats), v(0, stats)
        if n == 2:
            return __class__.pairwise_stats(
                counts[0], m(0, stats), v(0, stats),
                counts[1], m(1, stats), v(1, stats)
                )
        h = n // 2
        return __class__.pairwise_stats(
            *__class__.compute_stats(stats[:h], counts[:h], m, v),
            *__class__.compute_stats(stats[h:], counts[h:], m, v)
            )
