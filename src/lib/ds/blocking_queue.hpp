#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

template <class T>
class BlockingQueue {
   public:
    BlockingQueue() {}
    BlockingQueue(const BlockingQueue& other) = delete;
    BlockingQueue& operator=(const BlockingQueue& other) = delete;

    bool Enqueue(T elem) {
        if (terminate_ || sealed_) return false;
        std::unique_lock<std::mutex> lock{queue_mutex_};
        queue_.push_back(elem);
        if (queue_.size() == 1) {
            queue_cv_.notify_one();
        }
        return true;
    }

    bool Dequeue(T& elem) {
        std::unique_lock<std::mutex> lock{queue_mutex_};
        while (queue_.empty() && !terminate_ && !sealed_) {
            queue_cv_.wait(lock);
        }
        if (terminate_) return false;
        if (queue_.empty()) return false;
        elem = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

    void Terminate() {
        terminate_ = true;
        queue_cv_.notify_all();
    }

    void Seal() {
        sealed_ = true;
        queue_cv_.notify_all();
    }

   private:
    std::deque<T> queue_{};
    std::condition_variable queue_cv_{};
    std::mutex queue_mutex_{};
    std::atomic_bool terminate_{}, sealed_{};
};

template <class T>
class BlockingMulticastQueue {
   public:
    struct Subscription {
        bool Dequeue(T& elem) { return my_queue_->Dequeue(elem); }

        ~Subscription() {
            std::unique_lock<std::mutex> lock{ptr_->queue_mutex_};
            auto it = std::find_if(ptr_->queues_.begin(), ptr_->queues_.end(),
                                   [this](const auto& ptr) { return ptr.get() == my_queue_; });
            ptr_->queues_.erase(it);
        }

       private:
        Subscription(std::shared_ptr<BlockingMulticastQueue<T>> ptr) : ptr_{ptr} {
            std::unique_lock<std::mutex> lock{ptr_->queue_mutex_};
            my_queue_ = new BlockingQueue<T>();
            if (ptr_->terminate_) {
                my_queue_->Terminate();
            }
            if (ptr->sealed_) {
                my_queue_->Seal();
            }
            ptr_->queues_.emplace_back(my_queue_);
        }

        std::shared_ptr<BlockingMulticastQueue<T>> ptr_;
        BlockingQueue<T>* my_queue_;

        friend class BlockingMulticastQueue<T>;
    };

    friend class Subscription;

    static std::shared_ptr<BlockingMulticastQueue<T>> Create() {
        auto ptr = std::shared_ptr<BlockingMulticastQueue>(new BlockingMulticastQueue());
        ptr->self_ptr_ = ptr;
        return ptr;
    }

    bool Enqueue(T elem) {
        std::unique_lock<std::mutex> lock{queue_mutex_};
        bool success{true};
        for (auto& q : queues_) {
            success &= q->Enqueue(elem);
        }
        return success;
    }

    Subscription Subscribe() {
        auto ptr = self_ptr_.lock();
        if (!ptr) {
            throw std::runtime_error{"Cannot subscribe"};
        }
        return Subscription(ptr);
    }

    void Terminate() {
        terminate_ = true;
        std::unique_lock<std::mutex> lock{queue_mutex_};
        for (auto& q : queues_) {
            q->Terminate();
        }
    }

    void Seal() {
        sealed_ = true;
        std::unique_lock<std::mutex> lock{queue_mutex_};
        for (auto& q : queues_) {
            q->Seal();
        }
    }

    ~BlockingMulticastQueue() {}

   private:
    BlockingMulticastQueue() {}
    std::vector<std::unique_ptr<BlockingQueue<T>>> queues_;
    std::mutex queue_mutex_{};
    std::weak_ptr<BlockingMulticastQueue<T>> self_ptr_{};
    std::atomic_bool terminate_{}, sealed_{};
};