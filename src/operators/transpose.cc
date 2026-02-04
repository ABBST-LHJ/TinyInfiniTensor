#include "operators/transpose.h"
namespace infini
{
    TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                               vector<int> permute)
        : OperatorObj(OpType::Transpose, {input}, {output})
    {
        auto rank = input->getRank();
        if (permute.empty())
        {
            // 若 permute 为空，默认逆序排列所有维度（ONNX 规范）
            transposePermute.resize(rank);
            for (size_t i = 0; i < rank; ++i)
            {
                transposePermute[i] = rank - 1 - i;
            }
        }
        else
        {
            IT_ASSERT(rank == permute.size(), "permute 长度必须与输入张量维度数一致");
            transposePermute = std::move(permute);
        }
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        auto input_dim = A->getDims();
        int rank = A->getRank();
        auto output_dim = input_dim; // 初始化输出形状为输入形状

        // =================================== 作业实现 ===================================
        // 1. 验证 permute 合法性（维度索引在 [0, rank-1] 范围内）
        for (int p : transposePermute)
        {
            IT_ASSERT(p >= 0 && p < rank, "permute 包含无效维度索引");
        }

        // 2. 按 permute 顺序重新排列输入维度，得到输出形状
        for (int i = 0; i < rank; ++i)
        {
            int srcDimIdx = transposePermute[i]; // 输入维度的索引
            output_dim[i] = input_dim[srcDimIdx]; // 按 permute 映射维度大小
        }
        // =================================== 作业实现 ===================================

        return {{output_dim}}; // 返回推导后的输出形状（vector<Shape> 格式）
    }

    std::string TransposeObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "permute=" << vecToString(transposePermute) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }
}; // namespace infini
