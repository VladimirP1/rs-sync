#pragma once
#include <atomic>
#include <string>
#include <memory>

#include <ds/blocking_queue.hpp>

class Message {
   public:
    Message() {}
    Message(const Message&) = delete;
    Message(Message&&) = delete;
    Message& operator=(const Message&) = delete;
    Message& operator=(Message&&) = delete;

    virtual bool Take() {
        bool old = false;
        return taken_.compare_exchange_strong(old, true);
    }

    virtual bool Taken() const { return taken_; }

    virtual std::string ToString() const { return typeid(*this).name(); };
    virtual ~Message() {}

   private:
    std::atomic_bool taken_{};
};

class TaskMessage : public Message {};

class FrameLoaderTaskMessage : public TaskMessage {};
class KeypointDetectorTaskMessage : public TaskMessage {};


class EventMessage : public Message {};

class FrameLoaderEventMessage : public EventMessage {};
class KeypointDetectorEventMessage : public EventMessage {};

using MessageQueueT = BlockingMulticastQueue<std::shared_ptr<Message>>;
using MessageQueuePtr = std::shared_ptr<MessageQueueT>;