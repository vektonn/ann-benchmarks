import cProfile
from time import sleep, perf_counter
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy
from sklearn.preprocessing import normalize
from vektonn import Vektonn as VektonnClient
from vektonn.dtos import Attribute, AttributeValue, Vector, InputDataPoint, SearchQuery

from ann_benchmarks.algorithms.base import BaseANN

UPLOAD_BATCH_SIZE = 10000
VEKTONN_API_URL = 'http://vektonn-api:8081'
VEKTONN_INDEX_SHARD_URL = 'http://vektonn-index-shard:8082/api/v1/info'


def split_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


class Vektonn(BaseANN):
    def __init__(self, metric):
        if metric not in ('angular', 'euclidean'):
            raise NotImplementedError(f"Vektonn doesn't support metric {metric}")
        self._should_normalize = metric == 'angular'
        self._vektonn_client = VektonnClient(VEKTONN_API_URL)
        self.wait_for_vektonn(service_name='API', url=VEKTONN_API_URL)
        self.wait_for_vektonn(service_name='IndexShard', url=VEKTONN_INDEX_SHARD_URL)

    def fit(self, X):
        self.fit_impl(X)
        self.wait_for_vektonn_index_shard_is_initialized(expected_points_count=len(X))

    def fit_with_profiler(self, X):
        with cProfile.Profile() as pr:
            self.fit_impl(X)
        pr.print_stats(sort='time')

    def fit_impl(self, X):
        input_data_points = self.prepare_input_data_points(X)
        total_loaded_points = 0
        for batch in split_list(input_data_points, batch_size=UPLOAD_BATCH_SIZE):
            loaded_points, upload_time = self.upload_to_vektonn(batch)
            total_loaded_points += loaded_points
            # print(f"DEBUG: upload_to_vektonn({len(batch)}) took {upload_time}, total_loaded: {total_loaded_points}")

    def prepare_input_data_points(self, X):
        if self._should_normalize:
            X = normalize(X, axis=1, norm='l2')
            print("Normalized X for angular metric")

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)
            print("Converted X to numpy.float32")

        input_data_points = []
        for ind, vec in enumerate(X):
            input_data_points.append(self.to_idp(ind, vec))

        return input_data_points

    @staticmethod
    def to_idp(ind, vec):
        return InputDataPoint.construct(
            attributes=[Attribute.construct(key='id', value=AttributeValue.construct(int64=ind))],
            vector=Vector.construct(is_sparse=False, coordinates=vec.tolist()))

    def upload_to_vektonn(self, input_data_points):
        upload_start = perf_counter()
        self._vektonn_client.upload(
            data_source_name='ann-benchmark.source',
            data_source_version='1.0',
            input_data_points=input_data_points)
        upload_time = (perf_counter() - upload_start)
        return len(input_data_points), upload_time

    def query(self, v, n):
        query_start = perf_counter()

        if self._should_normalize:
            v /= numpy.linalg.norm(v)

        query = SearchQuery.construct(k=n, retrieveVectors=False, query_vectors=[
            Vector.construct(is_sparse=False, coordinates=v.tolist())
        ])
        search_results, search_time = self.search_vektonn(query)
        # print(f"DEBUG: search_vektonn(k={query.k}, vectors_len={len(query.query_vectors)}) took: {search_time}")
        candidates = self.get_candidates(search_results[0])

        query_time = (perf_counter() - query_start)
        # print(f"DEBUG: query() took: {query_time}")

        return candidates

    def search_vektonn(self, search_query):
        search_start = perf_counter()
        search_results = self._vektonn_client.search(
            index_name='ann-benchmark.index',
            index_version='1.0',
            search_query=search_query,
            request_timeout_seconds=600)
        search_time = (perf_counter() - search_start)
        return search_results, search_time

    def batch_query(self, X, n):
        self.batch_query_impl(X, n)

    def batch_query_with_profiler(self, X, n):
        with cProfile.Profile() as pr:
            self.batch_query_impl(X, n)
        pr.print_stats(sort='time')

    def batch_query_impl(self, X, n):
        if self._should_normalize:
            X /= numpy.linalg.norm(X)

        query = SearchQuery.construct(k=n, retrieveVectors=False, query_vectors=[
            Vector.construct(is_sparse=False, coordinates=vec.tolist()) for vec in X])
        self._search_results, search_time = self.search_vektonn(query)
        print(f"DEBUG: search_vektonn(k={query.k}, vectors_len={len(query.query_vectors)}) took: {search_time}")

    def get_batch_results(self):
        return [self.get_candidates(search_results)
                for search_results in self._search_results]

    @staticmethod
    def get_candidates(search_results):
        return [
            fdp.attributes[0].value.int64  # note that we have single attribute 'id'
            for fdp in search_results.nearest_data_points
        ]

    @staticmethod
    def wait_for_vektonn(service_name, url):
        print(f"Waiting for Vektonn {service_name} to start...")
        req = Request(url)
        for i in range(10):
            try:
                with urlopen(req) as response:
                    if response.getcode() == 200:
                        print(f"Vektonn {service_name} is ready")
                        return
            except URLError as exc:
                print(f"Vektonn waiting error: {exc}")
                pass
            sleep(3)
        raise RuntimeError(f"Failed to connect to Vektonn {service_name}")

    @staticmethod
    def wait_for_vektonn_index_shard_is_initialized(expected_points_count):
        print(f"Waiting for Vektonn IndexShard to contain {expected_points_count} data points...")
        req = Request(VEKTONN_INDEX_SHARD_URL)
        for i in range(600):
            with urlopen(req) as response:
                info = response.read().decode('utf-8')
                print(info)
                if f'"dataPointsCount":{expected_points_count}' in info:
                    return
            sleep(10)
        raise RuntimeError(f"Vektonn IndexShard does not contain {expected_points_count} data points")


class VektonnBruteForce(Vektonn):
    def __init__(self, metric, dimension):
        super().__init__(metric)
        self.name = 'VektonnBruteForce()'


class VektonnHnsw(Vektonn):
    def __init__(self, metric, dimension, hnsw_params):
        super().__init__(metric)
        self.name = f'VektonnHnsw({hnsw_params})'
