#pragma once
#include <atomic>
#include <string>
#include <memory>

#include <ds/blocking_queue.hpp>
#include <bl/component.hpp>

namespace rssync {

class Message {
   public:
    Message(const Message&) = delete;
    Message(Message&&) = delete;
    Message& operator=(const Message&) = delete;
    Message& operator=(Message&&) = delete;

    virtual std::string ToString() const { return typeid(*this).name(); }

    virtual std::string ReplyTo() final { return reply_to_; }

    virtual ~Message() {}

   protected:
    Message() {}

   private:
    std::string reply_to_;

    template <class T, class... Args>
    friend std::shared_ptr<Message> MakeMessage(std::string reply_to, Args&&... args);
};

class TaskMessage : public Message {};

class FrameLoaderTaskMessage : public TaskMessage {};
class KeypointDetectorTaskMessage : public TaskMessage {};

class EventMessage : public Message {};

class FrameLoaderEventMessage : public EventMessage {};
class KeypointDetectorEventMessage : public EventMessage {};

template <class T, class... Args>
std::shared_ptr<Message> MakeMessage(std::string reply_to, Args&&... args) {
    auto ptr = std::shared_ptr<T>(new T(std::forward<Args>(args)...));
    ptr->reply_to_ = reply_to;
    return ptr;
}

}  // namespace rssync