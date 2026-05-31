#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace rtxllm {

struct KvPage {
  std::uint32_t id = 0;
  bool used = false;
  std::uint32_t ref_count = 0;
};

class KvPageCache {
 public:
  KvPageCache() = default;

  KvPageCache(std::uint32_t page_count, std::size_t page_bytes)
      : page_bytes_(page_bytes), pages_(page_count) {
    if (page_count == 0 || page_bytes == 0) {
      throw std::invalid_argument("KV page cache requires non-zero page count and size");
    }
    for (std::uint32_t i = 0; i < page_count; ++i) {
      pages_[i].id = i;
    }
  }

  [[nodiscard]] std::optional<std::uint32_t> allocate() {
    for (auto& page : pages_) {
      if (!page.used) {
        page.used = true;
        page.ref_count = 1;
        ++used_pages_;
        return page.id;
      }
    }
    return std::nullopt;
  }

  void retain(std::uint32_t id) {
    auto& page = get(id);
    if (!page.used) {
      throw std::logic_error("cannot retain unused KV page");
    }
    ++page.ref_count;
  }

  void release(std::uint32_t id) {
    auto& page = get(id);
    if (!page.used || page.ref_count == 0) {
      throw std::logic_error("cannot release unused KV page");
    }
    --page.ref_count;
    if (page.ref_count == 0) {
      page.used = false;
      --used_pages_;
    }
  }

  [[nodiscard]] std::size_t page_bytes() const { return page_bytes_; }
  [[nodiscard]] std::size_t page_count() const { return pages_.size(); }
  [[nodiscard]] std::size_t used_pages() const { return used_pages_; }
  [[nodiscard]] double utilization() const {
    return pages_.empty() ? 0.0 : static_cast<double>(used_pages_) / pages_.size();
  }

 private:
  KvPage& get(std::uint32_t id) {
    if (id >= pages_.size()) {
      throw std::out_of_range("KV page id out of range");
    }
    return pages_[id];
  }

  std::size_t page_bytes_ = 0;
  std::vector<KvPage> pages_;
  std::size_t used_pages_ = 0;
};

}  // namespace rtxllm
