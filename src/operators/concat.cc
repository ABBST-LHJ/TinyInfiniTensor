#include "operators/concat.h"
#include "utils/operator_utils.h"
namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    IT_ASSERT(!inputs.empty(), "Concat 算子至少需要一个输入张量");
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    IT_ASSERT(!inputs.empty(), "Concat 算子至少需要一个输入张量");
    Shape dims = inputs[0]->getDims();
    size_t rank = static_cast<size_t>(inputs[0]->getRank()); // 统一为 size_t 类型

    // =================================== 作业实现 ===================================
    // 1. 校验所有输入张量的合法性（维度数一致 + 非拼接维度大小一致）
    for (const auto &tensor : inputs) {
        // 统一类型为 size_t 比较，避免符号差异
        size_t tensorRank = static_cast<size_t>(tensor->getRank());
        IT_ASSERT(tensorRank == rank, 
                  "输入张量维度数不一致：" + std::to_string(tensorRank) + 
                  " vs " + std::to_string(rank));
        
        const auto &tensorDims = tensor->getDims();
        // 循环变量为 size_t，与 rank 类型一致
        for (size_t i = 0; i < rank; ++i) {
            // 将 dim（int）强转为 size_t，避免符号比较
            if (i != static_cast<size_t>(dim) && tensorDims[i] != dims[i]) {
                IT_ASSERT(false, 
                          "非拼接维度大小不匹配：维度 " + std::to_string(i) + 
                          " 对应值 " + std::to_string(tensorDims[i]) + 
                          " vs " + std::to_string(dims[i]));
            }
        }
    }

    // 2. 计算拼接维度的总大小（累加所有输入张量的拼接维度大小）
    size_t concatDimTotal = 0;
    for (const auto &tensor : inputs) {
        // 拼接维度 dim 强转为 size_t，避免越界
        concatDimTotal += tensor->getDims()[static_cast<size_t>(dim)];
    }
    // 更新拼接维度大小（确保类型匹配 Shape 元素类型 int）
    dims[dim] = static_cast<int>(concatDimTotal);
    // =================================== 作业实现 ===================================

    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (size_t i = 0; i < inputs.size(); ++i) {
        os << vecToString(inputs[i]->getDims());
        if (i < inputs.size() - 1) os << ",";
    }
    os << ", dim=" << dim << ",";
    os << "input=[";
    for (size_t i = 0; i < inputs.size(); ++i) {
        os << inputs[i]->getGuid();
        if (i < inputs.size() - 1) os << ",";
    }
    os << "], output=" << outputs[0]->getGuid() << ")";
    return os.str();
}
} // namespace infini
