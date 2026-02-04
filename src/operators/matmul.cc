#include "operators/matmul.h"
#include "utils/operator_utils.h" // 用于 infer_broadcast 广播批量维度
namespace infini
{
    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B")
           << "],A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业实现 ===================================
        IT_ASSERT(inputs.size() == 2, "Matmul 仅支持两个输入张量");
        const auto A = inputs[0];
        const auto B = inputs[1];
        auto shapeA = A->getDims();
        auto shapeB = B->getDims();
        int rankA = shapeA.size();
        int rankB = shapeB.size();

        // 1. 处理矩阵维度（最后两维）：根据 transA/transB 调整实际矩阵维度
        // A 的矩阵维度：默认 (M, K)，transA=true 则转为 (K, M)
        int dimA1 = (rankA >= 1) ? shapeA.back() : 1;
        int dimA2 = (rankA >= 2) ? shapeA[rankA - 2] : 1;
        int matA_M = transA ? dimA1 : dimA2; // A 实际行数
        int matA_K = transA ? dimA2 : dimA1; // A 实际列数

        // B 的矩阵维度：默认 (K, N)，transB=true 则转为 (N, K)
        int dimB1 = (rankB >= 1) ? shapeB.back() : 1;
        int dimB2 = (rankB >= 2) ? shapeB[rankB - 2] : 1;
        int matB_K = transB ? dimB1 : dimB2; // B 实际行数
        int matB_N = transB ? dimB2 : dimB1; // B 实际列数

        // 校验矩阵维度兼容性：A 的列数必须等于 B 的行数
        IT_ASSERT(matA_K == matB_K, 
                  "Matmul 维度不兼容：A 的列数 " + std::to_string(matA_K) + 
                  " != B 的行数 " + std::to_string(matB_K));

        // 2. 处理批量维度（前 N-2 维）：对 A 和 B 的批量维度进行广播
        Shape batchA = (rankA >= 2) ? Shape(shapeA.begin(), shapeA.end() - 2) : Shape();
        Shape batchB = (rankB >= 2) ? Shape(shapeB.begin(), shapeB.end() - 2) : Shape();
        Shape broadcastedBatch = infer_broadcast(batchA, batchB);

        // 3. 构造输出形状：广播后的批量维度 + 矩阵相乘结果维度 (M, N)
        Shape outputShape = broadcastedBatch;
        outputShape.push_back(matA_M); // 输出矩阵行数 = A 实际行数
        outputShape.push_back(matB_N); // 输出矩阵列数 = B 实际列数

        // 4. 更新辅助属性 m/n/k（用于 toString 打印）
        m = matA_M;
        n = matB_N;
        k = matA_K;

        return {{outputShape}};
        // =================================== 作业实现 ===================================
    }
} // namespace infini
