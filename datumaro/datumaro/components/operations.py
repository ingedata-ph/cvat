
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum
import logging as log
from functools import reduce
from itertools import chain, zip_longest

import cv2
import numpy as np

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.config import Config, SchemaBuilder
from datumaro.components.extractor import AnnotationType, Bbox, LabelCategories
from datumaro.components.project import Dataset
from datumaro.util import find, pairs
from datumaro.util.annotation_util import segment_iou, bbox_iou, mean_bbox, OKS, PDJ, find_instances, max_bbox

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


class Configurable:
    def __init__(self, **options):
        self.__dict__['_conf'] = {}
        self.__dict__['_conf'] = Config(options,
            fallback=getattr(self, 'CONFIG_DEFAULTS', None),
            schema=getattr(self, 'CONFIG_SCHEMA', None),
            mutable=False)
        print(self._conf)

    def __getattr__(self, name):
        try:
            return self.__dict__['_conf'][name]
        except KeyError:
            raise AttributeError(name)

class MergingStrategy(CliPlugin, Configurable):
    @classmethod
    def merge(cls, sources, **options):
        instance = cls(**options)
        return instance(sources)

    def __init__(self, **options):
        super().__init__(**options)
        self.__dict__['_sources'] = None

    def __call__(self, sources):
        raise NotImplementedError()

class IntersectMerge(MergingStrategy):
    # TODO: put this class to the right place
    CONFIG_SCHEMA = SchemaBuilder() \
        .add('pairwise_dist', float) \
        .add('cluster_dist', float) \
        .add('do_nms', bool) \
        .add('input_conf_thresh', float) \
        .add('output_conf_thresh', float) \
        .add('quorum', int) \
        .add('ignored_attributes', set) \
        .add('metric', str) \
        .add('pdj_radius', float) \
        .add('pdj_ratio', float) \
        .add('oks_sigma', list) \
        \
        .add('check_lonely_clusters', bool) \
        .add('close_clusters_dist', float) \
        \
        .build()

    CONFIG_DEFAULTS = Config({
        'metric': 'OKS',
        'pdj_ratio': 0.05,
        'oks_sigma': [],
        'check_lonely_clusters': False,
        'close_clusters_dist': 0.5,
    })

    _image = None # TODO: remove after debug
    _item = None # TODO: remove after debug
    def __call__(self, datasets):
        merged = Dataset(
            categories=merge_categories(d.categories() for d in datasets))

        item_ids = set((item.id, item.subset) for d in datasets for item in d)

        for (item_id, item_subset) in sorted(item_ids, key=lambda e: e[0]):
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
        # print(self._item.id, self._item.image._data)
        self._image = self._item.image.data.copy()

        sources = [[a for a in item.annotations
            if self.input_conf_thresh <= a.attributes.get('score', 1)
            ] for item in items]
        if self.do_nms:
            sources = list(map(nms, sources))
        print(self._item.id, list(map(len, sources)))

        annotations = self.merge_annotations(sources)

        annotations = [a for a in annotations
            if self.output_conf_thresh <= a.attributes.get('score', 1)]

        from datumaro.util.image import save_image
        save_image('test_clusters/%s.jpg' % self._item.id, self._image,
            create_dir=True)

        return items[0].wrap(annotations=annotations)

    def merge_annotations(self, sources):
        self._instance_map = {}
        for s in sources:
            s_instances = find_instances(s)
            for inst in s_instances:
                inst_bbox = max_bbox(inst)
                for ann in inst:
                    self._instance_map[id(ann)] = [inst, inst_bbox]

        all_by_type = {}
        for s in sources:
            src_by_type = {}
            for a in s:
                src_by_type.setdefault(a.type, []).append(a)
            for k, v in src_by_type.items():
                all_by_type.setdefault(k, []).append(v)

        mergers = {}
        annotations = []
        clusters = {}
        for k, v in all_by_type.items():
            m = self._make_ann_merger(k)
            mergers[k] = m
            clusters.setdefault(k, []).extend(m.find_clusters(v)[0])

        joined_clusters = sum(clusters.values(), [])
        group_map = self._find_cluster_groups(joined_clusters)

        for k, v in clusters.items():
            annotations += mergers[k].merge_clusters(v, group_map)

        return annotations

    def _make_ann_merger(self, t):
        def _make(c):
            s = c(**{k: v for k, v in self._conf.items() if k in c.CONFIG_SCHEMA})
            s._image = self._image
            s._instance_map = self._instance_map
            return s

        if t is AnnotationType.label:
            return _make(LabelMergingStrategy)
        elif t is AnnotationType.bbox:
            return _make(BboxMergingStrategy)
        elif t is AnnotationType.mask:
            return _make(MaskMergingStrategy)
        elif t is AnnotationType.polygon:
            return _make(PolygonMergingStrategy)
        elif t is AnnotationType.polyline:
            return _make(LineMergingStrategy)
        elif t is AnnotationType.points:
            return _make(PointsMergingStrategy)
        elif t is AnnotationType.caption:
            return reduce(merge_annotations_unique, ann, [])
        else:
            raise NotImplementedError("Merge for %s type is not supported" % t)

    @staticmethod
    def _find_cluster_groups(clusters):
        cluster_groups = []
        visited = set()
        for a_idx, cluster_a in enumerate(clusters):
            if a_idx in visited:
                continue
            visited.add(a_idx)

            cluster_group = { id(cluster_a) }

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
                    cluster_group.add( id(cluster_b) )
                    visited.add(b_idx)

            cluster_groups.append( (cluster_group, a_groups) )
        return cluster_groups

class LabelMergingStrategy(Configurable):
    CONFIG_SCHEMA = SchemaBuilder() \
        .add('quorum', int) \
        .build()

    def merge_clusters(self, clusters):
        votes = {} # label -> score
        for s in chain(*clusters):
            for label_ann in s:
                votes[label_ann.label] = 1.0 + votes.get(value, 0.0)

        labels = {}
        for name, votes in votes.items():
            votes_count, label = max(votes.items(), key=lambda e: e[1])
            if self.quorum <= votes_count:
                labels[name] = label

        return labels

    @staticmethod
    def find_clusters(sources):
        return sum(sources)

class ShapeMergingStrategy(Configurable):
    CONFIG_SCHEMA = SchemaBuilder() \
        .add('pairwise_dist', float) \
        .add('cluster_dist', float) \
        .add('do_nms', bool) \
        .add('input_conf_thresh', float) \
        .add('output_conf_thresh', float) \
        .add('quorum', int) \
        .add('ignored_attributes', set) \
        \
        .add('check_lonely_clusters', bool) \
        .add('close_clusters_dist', float) \
        .build()

    def merge_clusters(self, clusters, groups):
        merged = []
        for cluster in clusters:
            label, label_score = self.find_cluster_label(cluster)
            shape, shape_score = self.merge_cluster_shape(cluster)

            attributes = self.find_cluster_attrs(cluster)
            attributes = { k: v for k, v in attributes.items()
                if k not in self.ignored_attributes }

            attributes['score'] = label_score * shape_score \
                if label is not None else shape_score

            new_group_id, (_, cluster_groups) = find(enumerate(groups),
                lambda e: id(cluster) in e[1][0])
            if not cluster_groups or cluster_groups == {0}:
                new_group_id = 0
            else:
                new_group_id += 1

            shape.label = label
            shape.group = new_group_id
            shape.attributes.update(attributes)

            # debugging code
            bbox = shape.get_bbox()
            cv2.rectangle(self._image, (int(bbox[0]), int(bbox[1])),
                (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                (255, 255, 255), thickness=1)

            merged.append(shape)

        return merged

    def find_cluster_label(self, cluster):
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
                matches, mismatches, xa, xb = compare_segments(src_a, src_b,
                    dist_thresh=pairwise_dist, distance=distance)
                print('matches', len(matches), 'mismatches', len(mismatches), 'xa', len(xa), 'xb', len(xb)) # debug
                for m in matches:
                    adjacent[id(m[0])].append(id(m[1]))

                # debugging code
                # mc = (0, 255, 0)
                # mmc = (0, 0, 255)
                # xc = (255, 0, 0)
                # for (ma, mb) in matches:
                #     b1 = ma.get_bbox()
                #     b2 = mb.get_bbox()
                #     cv2.line(self._image, (int(b1[0]), int(b1[1])),
                #         ((int(b2[0]), int(b2[1]))), mc,
                #         thickness=4)
                # for (ma, mb) in mismatches:
                #     b1 = ma.get_bbox()
                #     b2 = mb.get_bbox()
                #     cv2.line(self._image, (int(b1[0]), int(b1[1])),
                #         ((int(b2[0]), int(b2[1]))), mmc,
                #         thickness=4)
                #     cv2.putText(self._image, '%.3f' % self.distance(ma, mb),
                #         (int(b1[0]), int(b1[1])),
                #         cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), bottomLeftOrigin=True)
                # for ma in xa + xb:
                #     b1 = ma.get_bbox()
                #     cv2.circle(self._image, (int(b1[0]), int(b1[1])),
                #         5, xc,
                #         thickness=4)

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
        mbbox = Bbox(*mean_bbox(cluster))
        dist = (self.distance(mbbox, s) for s in cluster)
        min_dist_pos, _ = min(enumerate(dist), key=lambda e: e[1])
        return cluster[min_dist_pos]

    def merge_cluster_shape(self, cluster):
        shape = self._merge_cluster_shape_mean_box_nearest(cluster)
        shape_score = sum(max(0, self.distance(shape, s))
            for s in cluster) / len(cluster)
        return shape, shape_score

class BboxMergingStrategy(ShapeMergingStrategy):
    def find_clusters(self, sources):
        clusters, id_segm = super().find_clusters(sources)

        # debugging code
        # import datumaro.util.mask_tools as mask_tools
        # cm = mask_tools.generate_colormap()
        # for c_id, cluster in enumerate(clusters):
        #     color = tuple(map(int, cm[1 + c_id][::-1]))
        #     for segm in cluster:
        #         _, src_id = id_segm[id(segm)]
        #         cv2.rectangle(self._image, (int(segm.x), int(segm.y)),
        #             ((int(segm.points[2]), int(segm.points[3]))), color,
        #             thickness=2)
        #         cv2.putText(self._image, '(c%s,s%s)' % (c_id, src_id),
        #             (int(segm.x) + 1, int(segm.y) + 1),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
        #         cv2.putText(self._image, '(c%s,s%s)' % (c_id, src_id),
        #             (int(segm.x), int(segm.y)),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, color)

        return clusters, id_segm

class PolygonMergingStrategy(ShapeMergingStrategy):
    def find_clusters(self, sources):
        clusters, id_segm = super().find_clusters(sources)

        # debugging code
        import datumaro.util.mask_tools as mask_tools
        cm = mask_tools.generate_colormap()
        for c_id, cluster in enumerate(clusters):
            color = tuple(map(int, cm[1 + c_id][::-1]))
            for segm in cluster:
                _, src_id = id_segm[id(segm)]
                pts = np.array(segm.points).reshape((-1, 2))
                for p1, p2 in pairs(pts):
                    cv2.circle(self._image, (int(p1[0]), int(p1[1])), 1, color,
                        thickness=2)
                    cv2.line(self._image, (int(p1[0]), int(p1[1])),
                        (int(p2[0]), int(p2[1])), color, thickness=2)
                bbox = segm.get_bbox()
                cv2.putText(self._image, '(c%s,s%s)' % (c_id, src_id),
                    (int(bbox[0]) + 1, int(bbox[1]) + 1),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
                cv2.putText(self._image, '(c%s,s%s)' % (c_id, src_id),
                    (int(bbox[0]), int(bbox[1])),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, color)

        return clusters, id_segm

class MaskMergingStrategy(ShapeMergingStrategy):
    def find_clusters(self, sources):
        clusters, id_segm = super().find_clusters(sources)

        # debugging code
        import datumaro.util.mask_tools as mask_tools
        cm = mask_tools.generate_colormap()
        for c_id, cluster in enumerate(clusters):
            color = tuple(map(int, cm[1 + c_id][::-1]))
            for segm in cluster:
                _, src_id = id_segm[id(segm)]
                for p in np.array(segm.points).reshape((-1, 2)):
                    cv2.circle(self._image, (int(p[0]), int(p[1])), 1, color,
                        thickness=2)
                bbox = segm.get_bbox()
                cv2.putText(self._image, '(c%s,s%s)' % (c_id, src_id),
                    (int(bbox[0]) + 1, int(bbox[1]) + 1),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
                cv2.putText(self._image, '(c%s,s%s)' % (c_id, src_id),
                    (int(bbox[0]), int(bbox[1])),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, color)

        return clusters, id_segm

Metrics = Enum('Metrics', ['OKS', 'PDJ'])
class PointsMergingStrategy(ShapeMergingStrategy):
    CONFIG_SCHEMA = SchemaBuilder(fallback=ShapeMergingStrategy.CONFIG_SCHEMA) \
        .add('metric', str) \
        .add('pdj_radius', float) \
        .add('pdj_ratio', float) \
        .add('oks_sigma', list) \
        .add('image_size', tuple) \
        .build()

    def distance(self, a, b):
        a_bbox = self._instance_map[id(a)][1]
        b_bbox = self._instance_map[id(b)][1]
        if bbox_iou(a_bbox, b_bbox) <= 0:
            return 0
        bbox = mean_bbox([a_bbox, b_bbox])
        if Metrics[self.metric] == Metrics.PDJ:
            return PDJ(a, b, eps=self.pdj_radius, ratio=self.pdj_ratio, bbox=bbox)
        elif Metrics[self.metric] == Metrics.OKS:
            return OKS(a, b, sigma=self.oks_sigma, bbox=bbox)
        else:
            raise NotImplementedError("Unknown metric type '%s'" % self.metric)

    def find_clusters(self, sources):
        clusters, id_segm = super().find_clusters(sources)

        # debugging code
        # import datumaro.util.mask_tools as mask_tools
        # cm = mask_tools.generate_colormap()
        # for c_id, cluster in enumerate(clusters):
        #     color = tuple(map(int, cm[1 + c_id][::-1]))
        #     for segm in cluster:
        #         _, src_id = id_segm[id(segm)]
        #         for p in np.array(segm.points).reshape((-1, 2)):
        #             cv2.circle(self._image, (int(p[0]), int(p[1])), 1, color,
        #                 thickness=2)
        #         bbox = segm.get_bbox()
        #         cv2.putText(self._image, '(c%s,s%s)' % (c_id, src_id),
        #             (int(bbox[0]) + 1, int(bbox[1]) + 1),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
        #         cv2.putText(self._image, '(c%s,s%s)' % (c_id, src_id),
        #             (int(bbox[0]), int(bbox[1])),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, color)

        return clusters, id_segm

    def _merge_cluster_shape_mean_box_nearest(self, cluster):
        mbbox = Bbox(*mean_bbox(cluster))
        dist = (segment_iou(mbbox, s) for s in cluster)
        min_dist_pos, _ = min(enumerate(dist), key=lambda e: e[1])
        return cluster[min_dist_pos]

class LineMergingStrategy:
    @staticmethod
    def distance(a, b):
        point_count = max(max(len(a.points) // 2, len(b.points) // 2), 100)
        a = smooth_line(a.points, point_count)
        b = smooth_line(b.points, point_count)

        p1 = np.linalg.norm(a, axis=1)
        p1 /= np.linalg.norm(p1)

        p2 = np.linalg.norm(b, axis=1)
        p2 /= np.linalg.norm(p2)

        return abs(np.dot(p1, p2))

    def find_clusters(self, sources):
        clusters, id_segm = super().find_clusters(sources)

        # debugging code
        import datumaro.util.mask_tools as mask_tools
        cm = mask_tools.generate_colormap()
        for c_id, cluster in enumerate(clusters):
            color = tuple(map(int, cm[1 + c_id][::-1]))
            for segm in cluster:
                _, src_id = id_segm[id(segm)]
                for p in np.array(segm.points).reshape((-1, 2)):
                    cv2.circle(self._image, (int(p[0]), int(p[1])), 1, color,
                        thickness=2)
                bbox = segm.get_bbox()
                cv2.putText(self._image, '(c%s,s%s)' % (c_id, src_id),
                    (int(bbox[0]) + 1, int(bbox[1]) + 1),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
                cv2.putText(self._image, '(c%s,s%s)' % (c_id, src_id),
                    (int(bbox[0]), int(bbox[1])),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, color)

        return clusters, id_segm

    def merge_cluster_shape(self, cluster):
        shape = self._merge_cluster_shape_mean_box_nearest(cluster)
        shape_score = sum(max(0, self.distance(shape, s))
            for s in cluster) / len(cluster)
        return shape, shape_score

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
    # debugging code
    # if a_segms[0].type == AnnotationType.points:
        # bdistances = np.array([[distance(Bbox(*a.get_bbox()), Bbox(*b.get_bbox())) for b in b_segms] for a in a_segms])
        # print(a_segms[0], distances, dist_thresh)

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
