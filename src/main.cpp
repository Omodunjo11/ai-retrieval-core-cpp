#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <thread>
#include <cstdint>

#include "vector_search.h"

using namespace std;

constexpr size_t DIM = 768;
constexpr size_t NUM_VECTORS = 100000;
constexpr size_t TOP_K = 5;

int main() {
    cout << "Generating embeddings..." << endl;

    vector<float> database(NUM_VECTORS * DIM);
    vector<float> query(DIM);

    mt19937 rng(42);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& v : database) v = dist(rng);
    for (auto& v : query) v = dist(rng);

    // -----------------------------
    // Quantization
    // -----------------------------

    cout << "Quantizing database..." << endl;

    vector<int8_t> database_q(NUM_VECTORS * DIM);
    vector<float> scales(NUM_VECTORS);

    for (size_t i = 0; i < NUM_VECTORS; ++i) {
        quantize_vector(
            &database[i * DIM],
            &database_q[i * DIM],
            scales[i],
            DIM
        );
    }

    vector<int8_t> query_q(DIM);
    float query_scale;

    quantize_vector(
        query.data(),
        query_q.data(),
        query_scale,
        DIM
    );

    size_t num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    size_t chunk_size = NUM_VECTORS / num_threads;

    // =====================================================
    // FLOAT SEARCH
    // =====================================================

    cout << "\nRunning FLOAT search..." << endl;

    auto start_float = chrono::high_resolution_clock::now();

    vector<vector<pair<float, size_t>>> thread_results_float(num_threads);
    vector<thread> threads_float;

    for (size_t t = 0; t < num_threads; ++t) {
        threads_float.emplace_back([&, t]() {

            size_t start_idx = t * chunk_size;
            size_t end_idx = (t == num_threads - 1)
                                 ? NUM_VECTORS
                                 : start_idx + chunk_size;

            auto& local_heap = thread_results_float[t];
            local_heap.reserve(TOP_K);

            for (size_t i = start_idx; i < end_idx; ++i) {
                float score = dot_product(
                    &database[i * DIM],
                    query.data(),
                    DIM
                );

                if (local_heap.size() < TOP_K) {
                    local_heap.emplace_back(score, i);
                    if (local_heap.size() == TOP_K) {
                        make_heap(local_heap.begin(), local_heap.end(),
                            [](const auto& a, const auto& b) {
                                return a.first > b.first;
                            });
                    }
                } else if (score > local_heap.front().first) {
                    pop_heap(local_heap.begin(), local_heap.end(),
                        [](const auto& a, const auto& b) {
                            return a.first > b.first;
                        });
                    local_heap.back() = {score, i};
                    push_heap(local_heap.begin(), local_heap.end(),
                        [](const auto& a, const auto& b) {
                            return a.first > b.first;
                        });
                }
            }
        });
    }

    for (auto& th : threads_float) th.join();

    vector<pair<float, size_t>> final_heap_float;
    final_heap_float.reserve(TOP_K);

    for (auto& local : thread_results_float) {
        for (auto& elem : local) {

            if (final_heap_float.size() < TOP_K) {
                final_heap_float.push_back(elem);
                if (final_heap_float.size() == TOP_K) {
                    make_heap(final_heap_float.begin(), final_heap_float.end(),
                        [](const auto& a, const auto& b) {
                            return a.first > b.first;
                        });
                }
            } else if (elem.first > final_heap_float.front().first) {
                pop_heap(final_heap_float.begin(), final_heap_float.end(),
                    [](const auto& a, const auto& b) {
                        return a.first > b.first;
                    });
                final_heap_float.back() = elem;
                push_heap(final_heap_float.begin(), final_heap_float.end(),
                    [](const auto& a, const auto& b) {
                        return a.first > b.first;
                    });
            }
        }
    }

    auto end_float = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_float = end_float - start_float;

    cout << "FLOAT search time: "
         << elapsed_float.count()
         << " seconds\n";

    cout << "Top score (float): "
         << final_heap_float.front().first
         << endl;

    // =====================================================
    // INT8 SEARCH
    // =====================================================

    cout << "\nRunning INT8 search..." << endl;

    auto start_int8 = chrono::high_resolution_clock::now();

    vector<vector<pair<float, size_t>>> thread_results_int8(num_threads);
    vector<thread> threads_int8;

    for (size_t t = 0; t < num_threads; ++t) {
        threads_int8.emplace_back([&, t]() {

            size_t start_idx = t * chunk_size;
            size_t end_idx = (t == num_threads - 1)
                                 ? NUM_VECTORS
                                 : start_idx + chunk_size;

            auto& local_heap = thread_results_int8[t];
            local_heap.reserve(TOP_K);

            for (size_t i = start_idx; i < end_idx; ++i) {

                int32_t raw = dot_product_int8(
                    &database_q[i * DIM],
                    query_q.data(),
                    DIM
                );

                float score = raw * scales[i] * query_scale;

                if (local_heap.size() < TOP_K) {
                    local_heap.emplace_back(score, i);
                    if (local_heap.size() == TOP_K) {
                        make_heap(local_heap.begin(), local_heap.end(),
                            [](const auto& a, const auto& b) {
                                return a.first > b.first;
                            });
                    }
                } else if (score > local_heap.front().first) {
                    pop_heap(local_heap.begin(), local_heap.end(),
                        [](const auto& a, const auto& b) {
                            return a.first > b.first;
                        });
                    local_heap.back() = {score, i};
                    push_heap(local_heap.begin(), local_heap.end(),
                        [](const auto& a, const auto& b) {
                            return a.first > b.first;
                        });
                }
            }
        });
    }

    for (auto& th : threads_int8) th.join();

    vector<pair<float, size_t>> final_heap_int8;
    final_heap_int8.reserve(TOP_K);

    for (auto& local : thread_results_int8) {
        for (auto& elem : local) {

            if (final_heap_int8.size() < TOP_K) {
                final_heap_int8.push_back(elem);
                if (final_heap_int8.size() == TOP_K) {
                    make_heap(final_heap_int8.begin(), final_heap_int8.end(),
                        [](const auto& a, const auto& b) {
                            return a.first > b.first;
                        });
                }
            } else if (elem.first > final_heap_int8.front().first) {
                pop_heap(final_heap_int8.begin(), final_heap_int8.end(),
                    [](const auto& a, const auto& b) {
                        return a.first > b.first;
                    });
                final_heap_int8.back() = elem;
                push_heap(final_heap_int8.begin(), final_heap_int8.end(),
                    [](const auto& a, const auto& b) {
                        return a.first > b.first;
                    });
            }
        }
    }

    auto end_int8 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_int8 = end_int8 - start_int8;

    cout << "INT8 search time: "
         << elapsed_int8.count()
         << " seconds\n";

    cout << "Top score (int8): "
         << final_heap_int8.front().first
         << endl;

    // =====================================================
    // Memory + Bandwidth Reporting
    // =====================================================

    size_t float_bytes = NUM_VECTORS * DIM * sizeof(float);
    size_t int8_bytes  = NUM_VECTORS * DIM * sizeof(int8_t);

    cout << "\nMemory footprint:\n";
    cout << "Float DB: "
         << float_bytes / (1024.0 * 1024.0)
         << " MB\n";

    cout << "Int8 DB:  "
         << int8_bytes / (1024.0 * 1024.0)
         << " MB\n";

    cout << "Reduction: "
         << 100.0 * (1.0 - (double)int8_bytes / float_bytes)
         << "%\n";

    double bandwidth_float =
        float_bytes / elapsed_float.count() / 1e9;

    double bandwidth_int8 =
        int8_bytes / elapsed_int8.count() / 1e9;

    cout << "\nEstimated bandwidth (float): "
         << bandwidth_float
         << " GB/s\n";

    cout << "Estimated bandwidth (int8): "
         << bandwidth_int8
         << " GB/s\n";

    return 0;
}
