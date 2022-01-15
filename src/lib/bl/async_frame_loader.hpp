#pragma once
#include <memory>
#include <future>

#include <ds/sync_context.hpp>

class AsyncFrameLoader {
   public:
    static std::shared_ptr<AsyncFrameLoader> CreateFrameLoader(std::string path);
    
    virtual std::future<int> GetFrameNumberAt(double timestamp) = 0;
    virtual std::future<int> GetFrameCount() = 0;
    virtual std::future<void> LoadFramesIntoContext(int begin, int end, std::shared_ptr<SyncContext> ctx) = 0;
    virtual ~AsyncFrameLoader();
};