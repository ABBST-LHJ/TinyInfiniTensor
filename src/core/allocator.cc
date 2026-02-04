#include "core/allocator.h"
#include <utility>

namespace infini {
Allocator::Allocator(Runtime runtime) : runtime(runtime)
{
    used = 0;
    peak = 0;
    ptr = nullptr;
    alignment = sizeof(uint64_t);
    // 初始化空闲块：初始状态无空闲块（首次分配从 0 地址开始）
}

Allocator::~Allocator()
{
    if (this->ptr != nullptr)
    {
        runtime->dealloc(this->ptr);
    }
}

size_t Allocator::alloc(size_t size)
{
    IT_ASSERT(this->ptr == nullptr);
    size = this->getAlignedSize(size);
    if (size == 0)
        return 0;

    // =================================== 作业实现 ===================================
    size_t allocatedAddr = 0;

    // 1. 查找空闲块（首次适配：找第一个能容纳 size 的空闲块）
    auto it = freeBlocks.begin();
    for (; it != freeBlocks.end(); ++it)
    {
        size_t blockAddr = it->first;
        size_t blockSize = it->second;

        if (blockSize >= size)
        {
            // 找到合适的空闲块，分配该块的前 size 字节
            allocatedAddr = blockAddr;

            // 更新空闲块：如果剩余空间大于 0，保留剩余部分；否则删除该空闲块
            if (blockSize > size)
            {
                freeBlocks.erase(it);
                freeBlocks.emplace(blockAddr + size, blockSize - size);
            }
            else
            {
                freeBlocks.erase(it);
            }

            break;
        }
    }

    // 2. 无合适空闲块，从已用内存末尾分配
    if (it == freeBlocks.end())
    {
        allocatedAddr = used; // 新块起始地址 = 当前已用内存大小
    }

    // 3. 更新已用内存和峰值内存
    used += size;
    if (used > peak)
    {
        peak = used;
    }

    return allocatedAddr;
    // =================================== 作业实现 ===================================
}

void Allocator::free(size_t addr, size_t size)
{
    IT_ASSERT(this->ptr == nullptr);
    size = getAlignedSize(size);
    if (size == 0)
        return;

    // =================================== 作业实现 ===================================
    // 1. 新增空闲块（先不合并，后续处理）
    auto [newIt, inserted] = freeBlocks.emplace(addr, size);
    IT_ASSERT(inserted, "重复释放同一内存块");

    // 2. 合并相邻空闲块（前向合并 + 后向合并）
    size_t currentAddr = addr;
    size_t currentSize = size;

    // 前向合并：检查当前块的前一个块是否相邻（前块结束地址 = 当前块起始地址）
    auto prevIt = newIt;
    if (prevIt != freeBlocks.begin())
    {
        --prevIt;
        size_t prevAddr = prevIt->first;
        size_t prevSize = prevIt->second;

        if (prevAddr + prevSize == currentAddr)
        {
            // 合并前块和当前块
            currentAddr = prevAddr;
            currentSize += prevSize;
            freeBlocks.erase(prevIt);
            freeBlocks.erase(newIt);
            newIt = freeBlocks.emplace(currentAddr, currentSize).first;
        }
    }

    // 后向合并：检查当前块的后一个块是否相邻（当前块结束地址 = 后块起始地址）
    auto nextIt = newIt;
    ++nextIt;
    if (nextIt != freeBlocks.end())
    {
        size_t nextAddr = nextIt->first;
        size_t nextSize = nextIt->second;

        if (currentAddr + currentSize == nextAddr)
        {
            // 合并当前块和后块
            currentSize += nextSize;
            freeBlocks.erase(nextIt);
            freeBlocks.erase(newIt);
            freeBlocks.emplace(currentAddr, currentSize);
        }
    }

    // 3. 更新已用内存
    used -= size;
    // =================================== 作业实现 ===================================
}

void *Allocator::getPtr()
{
    if (this->ptr == nullptr)
    {
        this->ptr = runtime->alloc(this->peak);
        printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
    }
    return this->ptr;
}

size_t Allocator::getAlignedSize(size_t size)
{
    return ((size - 1) / this->alignment + 1) * this->alignment;
}

void Allocator::info()
{
    std::cout << "Used memory: " << this->used
              << ", peak memory: " << this->peak
              << ", free blocks count: " << freeBlocks.size() << std::endl;
}
}
