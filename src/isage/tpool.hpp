// #ifndef ISAGE_THREAD_POOL_H_
// #define ISAGE_THREAD_POOL_H_

// #include <vector>
// #include <queue>
// #include <memory>
// #include <thread>
// #include <mutex>
// #include <condition_variable>
// #include <future>
// #include <functional>
// #include <stdexcept>

// namespace isage {

//   class ThreadPool {
//   public:
//     ThreadPool(size_t num_workers);
//     ~ThreadPool();
  
//     template <typename Function, typename... Args>
//     void add(Function&& function, Args&&... args) {
//     }
//   private:
//     const size_t num_threads_max_;
//     std::vector< std::thread > 
//   };
// }

// #endif
