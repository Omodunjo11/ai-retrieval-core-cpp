#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

#include <thread>
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

    cout << "Running search..." << endl;

    auto start = chrono::high_resolution_clock::now();

    size_t num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    size_t chunk_size = NUM_VECTORS / num_threads;

    vector<vector<pair<float, size_t>>> thread_results(num_threads);
    vector<thread> threads;

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {

            size_t start_idx = t * chunk_size;
            size_t end_idx = (t == num_threads - 1) ? NUM_VECTORS : start_idx + chunk_size;

            auto& local_heap = thread_results[t];
            local_heap.reserve(TOP_K);

            for (size_t i = start_idx; i < end_idx; ++i) {
                float score = dot_product(&database[i * DIM], query.data(), DIM);

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

    for (auto& th : threads) {
        th.join();
    }

    // Merge thread heaps
    vector<pair<float, size_t>> final_heap;
    final_heap.reserve(TOP_K);

    for (auto& local : thread_results) {
        for (auto& elem : local) {

            if (final_heap.size() < TOP_K) {
                final_heap.push_back(elem);
                if (final_heap.size() == TOP_K) {
                    make_heap(final_heap.begin(), final_heap.end(),
                        [](const auto& a, const auto& b) {
                            return a.first > b.first;
                        });
                }
            } else if (elem.first > final_heap.front().first) {
                pop_heap(final_heap.begin(), final_heap.end(),
                    [](const auto& a, const auto& b) {
                        return a.first > b.first;
                    });
                final_heap.back() = elem;
                push_heap(final_heap.begin(), final_heap.end(),
                    [](const auto& a, const auto& b) {
                        return a.first > b.first;
                    });
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Search time: " << elapsed.count() << " seconds" << endl;
    cout << "Top score: " << final_heap.front().first << endl;

    return 0;
}