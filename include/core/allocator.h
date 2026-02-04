#pragma once
#include "core/runtime.h"
#include "core/tensor.h"
#ifdef BUILD_TEST
#include "gtest/gtest.h"
#endif
#include <cstddef>
#include <map>
#include <unordered_set>

namespace infini {
class Allocator
{
private:
    Runtime runtime;
    size_t used;
    size_t peak;
    size_t alignment;
    void *ptr;

    // =================================== 作业实现 ===================================
    // 空闲块管理：key=起始地址偏移量，value=块大小（已对齐）
    // map 按起始地址升序排列，便于查找和合并相邻块
    std::map<size_t, size_t> freeBlocks;
    // =================================== 作业实现 ===================================

public:
    Allocator(Runtime runtime);
    virtual ~Allocator();
    size_t alloc(size_t size);
    void free(size_t addr, size_t size);
    void *getPtr();
    void info();

private:
    size_t getAlignedSize(size_t size);
};
}
