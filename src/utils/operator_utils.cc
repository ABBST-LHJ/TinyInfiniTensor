#include "utils/operator_utils.h"
#include "core/runtime.h"
namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {
    // =================================== 作业实现 ===================================
    Shape broadcastedShape;
    size_t rankA = A.size();
    size_t rankB = B.size();
    size_t maxRank = std::max(rankA, rankB);

    // 从右往左对齐维度，处理每个维度的广播逻辑
    for (size_t i = 0; i < maxRank; ++i) {
        // 计算当前维度在 A 和 B 中的索引（从右往左）
        size_t idxA = rankA - 1 - i;
        size_t idxB = rankB - 1 - i;

        // 获取当前维度的大小（超出张量维度范围则视为 1）
        int dimA = (idxA < rankA) ? A[idxA] : 1;
        int dimB = (idxB < rankB) ? B[idxB] : 1;

        // 广播规则：维度大小为 1 可广播到另一维度大小，非 1 则必须相等
        IT_ASSERT(dimA == 1 || dimB == 1 || dimA == dimB,
                  "广播失败：维度大小不兼容（" + std::to_string(dimA) + " vs " + std::to_string(dimB) + "）");

        // 广播后的维度大小取两者中的较大值（或相等值）
        broadcastedShape.push_back(std::max(dimA, dimB));
    }

    // 由于是从右往左处理，结果需要反转回正确的维度顺序
    std::reverse(broadcastedShape.begin(), broadcastedShape.end());
    return broadcastedShape;
    // =================================== 作业实现 ===================================
}


int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
