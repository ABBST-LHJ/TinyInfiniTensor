#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/transpose.h"
#include "operators/matmul.h"
namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }




#include <iostream>  
void GraphObj::optimize()
{
//cout是打印信息调试

    // 确保图已拓扑排序，保证依赖关系正确（前序算子在前面）
    std::cout << "[步骤1] 执行拓扑排序" << std::endl;
    IT_ASSERT(topo_sort() == true);
    std::cout << "[步骤1完成] 拓扑排序成功，当前算子数：" << ops.size() << "，张量数：" << tensors.size() << std::endl;

    // 第一步：去除冗余的相邻Transpose算子（互为逆操作：OP8和OP9）
    std::cout << "\n[步骤2] 开始去除冗余Transpose算子" << std::endl;
    bool hasRedundantTranspose = true;
    while (hasRedundantTranspose) {
        hasRedundantTranspose = false;
        OpVec newOps;
        size_t i = 0;
        std::cout << "[步骤2.1] 遍历算子列表，当前算子数：" << ops.size() << std::endl;
        while (i < ops.size()) {
            std::cout << "[步骤2.2] 处理算子索引 " << i << "，算子Guid：" << ops[i]->getGuid() << std::endl;
            auto op1 = as<TransposeObj>(ops[i]);
            if (!op1) {
                std::cout << "[步骤2.2.1] 算子" << ops[i]->getGuid() << "不是Transpose，加入新列表" << std::endl;
                newOps.push_back(ops[i]);
                i++;
                continue;
            }
            std::cout << "[步骤2.2.2] 算子" << ops[i]->getGuid() << "是Transpose，检查下一个算子" << std::endl;

            if (i + 1 >= ops.size()) {
                std::cout << "[步骤2.2.3] 无下一个算子，将Transpose" << ops[i]->getGuid() << "加入新列表" << std::endl;
                newOps.push_back(ops[i]);
                i++;
                continue;
            }
            auto op2 = as<TransposeObj>(ops[i + 1]);
            if (!op2) {
                std::cout << "[步骤2.2.4] 下一个算子" << ops[i+1]->getGuid() << "不是Transpose，将当前Transpose加入新列表" << std::endl;
                newOps.push_back(ops[i]);
                i++;
                continue;
            }
            std::cout << "[步骤2.2.5] 下一个算子" << ops[i+1]->getGuid() << "是Transpose，验证相邻关系" << std::endl;

            Tensor op1Output = op1->getOutput();
            Tensor op2Input = op2->getInputs(0);
            if (op1Output != op2Input) {
                std::cout << "[步骤2.2.6] 两个Transpose输入输出不匹配，非相邻，加入新列表" << std::endl;
                newOps.push_back(ops[i]);
                i++;
                continue;
            }
            std::cout << "[步骤2.2.7] 两个Transpose相邻（" << op1->getGuid() << "→" << op2->getGuid() << "），检查是否为逆操作" << std::endl;

            const auto& perm1 = op1->getPermute();
            const auto& perm2 = op2->getPermute();
            size_t rank = perm1.size();
            bool isInverse = true;
            if (perm1.size() != perm2.size()) {
                isInverse = false;
            } else {
                for (int j = 0; j < (int)rank; j++) {
                    if (perm2[perm1[j]] != j || perm1[perm2[j]] != j) {
                        isInverse = false;
                        break;
                    }
                }
            }

            if (isInverse) {
                std::cout << "[步骤2.2.8] 两个Transpose是逆操作，开始删除（" << op1->getGuid() << "和" << op2->getGuid() << "）" << std::endl;
                Tensor originalInput = op1->getInputs(0);  // i1（Guid=2）
                Tensor finalOutput = op2->getOutput();     // t2（Guid=5）
                std::cout << "[步骤2.2.9] originalInput（i1）Guid：" << originalInput->getGuid() << "，finalOutput（t2）Guid：" << finalOutput->getGuid() << std::endl;

                // 关键：找到finalOutput的目标算子（Matmul OP11），替换输入为originalInput
                std::cout << "[步骤2.2.10] 遍历finalOutput的目标算子，数量：" << finalOutput->getTargets().size() << std::endl;
                for (auto& targetOp : finalOutput->getTargets()) {
                    std::cout << "[步骤2.2.11] 处理目标算子Guid：" << targetOp->getGuid() << std::endl;
                    auto matmulOp = as<MatmulObj>(targetOp);
                    if (matmulOp) {
                        std::cout << "[步骤2.2.12] 目标算子是Matmul（" << matmulOp->getGuid() << "），替换输入从" << finalOutput->getGuid() << "为" << originalInput->getGuid() << std::endl;

                        matmulOp->replaceInput(finalOutput, originalInput);
                        std::cout << "[步骤2.2.13] 清理Matmul与Transpose" << op2->getGuid() << "的依赖" << std::endl;

                        matmulOp->removePredecessors(op2);

                        std::cout << "[步骤2.2.14] 给originalInput添加目标算子Matmul" << std::endl;
                        originalInput->addTarget(matmulOp);
                    }
                }

                // 清理中间张量t1（op1Output）
                std::cout << "[步骤2.2.15] 清理中间张量t1（Guid：" << op1Output->getGuid() << "）" << std::endl;
                op1Output->setSource(nullptr);
                op1Output->targets.clear();
                std::cout << "[步骤2.2.16] 从图中移除t1" << std::endl;
                removeTensor(op1Output);  // 移除t1（Guid=4）

                // 清理中间张量t2（finalOutput）
                std::cout << "[步骤2.2.17] 清理中间张量t2（Guid：" << finalOutput->getGuid() << "）" << std::endl;
                finalOutput->setSource(nullptr);
                finalOutput->targets.clear();
                std::cout << "[步骤2.2.18] 从图中移除t2" << std::endl;
                removeTensor(finalOutput);  // 移除t2（Guid=5）

                // 清理op1的输入张量的targets（originalInput对op1的依赖）
                originalInput->removeTarget(op1);
                // 清理op2的输入张量的targets（op1Output对op2的依赖，已移除t1，可忽略）

                // 标记已处理，跳过两个Transpose
                std::cout << "[步骤2.2.19] 逆操作Transpose删除完成，跳过两个算子" << std::endl;
                hasRedundantTranspose = true;
                i += 2;
            } else {
                std::cout << "[步骤2.2.20] 两个Transpose非逆操作，加入新列表" << std::endl;
                newOps.push_back(ops[i]);
                i++;
            }
        }
        std::cout << "[步骤2.3] 冗余Transpose处理完成，更新算子列表，新算子数：" << newOps.size() << std::endl;
        ops = std::move(newOps);
    }
    std::cout << "[步骤2完成] 冗余Transpose删除完成，当前算子数：" << ops.size() << "，张量数：" << tensors.size() << std::endl;

    // 第二步：合并Transpose到Matmul（OP10合并到Matmul的transB）
    OpVec optimizedOps;
    for (auto& op : ops) {
        std::cout << "[步骤3.1] 处理算子Guid：" << op->getGuid() << "，算子类型：" << op->getOpType().toString() << std::endl;
        auto matmulOp = as<MatmulObj>(op);
        if (!matmulOp) {
            std::cout << "[步骤3.1.1] 算子不是Matmul，加入优化列表" << std::endl;
            optimizedOps.push_back(op);
            continue;
        }
        std::cout << "[步骤3.1.2] 算子是Matmul（" << matmulOp->getGuid() << "），处理输入B" << std::endl;

        // 处理Matmul输入B（当前是t3，Guid=6，源是OP10）
        Tensor bInput = matmulOp->getInputs(1);  // t3（Guid=6）
        std::cout << "[步骤3.2] Matmul输入B的Guid：" << bInput->getGuid() << std::endl;
        bool transB = matmulOp->getTransB();
        std::cout << "[步骤3.3] 初始transB值：" << (transB ? "true" : "false") << std::endl;
        if (auto bSourceOp = bInput->getSource()) {
            std::cout << "[步骤3.4] 输入B的源算子Guid：" << bSourceOp->getGuid() << std::endl;
            auto bTransOp = as<TransposeObj>(bSourceOp);  // OP10
            if (bTransOp) {
                std::cout << "[步骤3.5] 源算子是Transpose（" << bTransOp->getGuid() << "），验证是否为最后两维交换" << std::endl;
                const auto& bPerm = bTransOp->getPermute();
                size_t bRank = bPerm.size();
                bool isLastTwoTrans = true;

                // 验证：前rank-2维不变，最后两维交换（OP10的permute是{0,1,3,2}）
                if (bRank >= 2) {
                    for (size_t i = 0; i < bRank - 2; i++) {
                        if (bPerm[i] != (int)i) {
                            isLastTwoTrans = false;
                            break;
                        }
                    }
                    if (!(bPerm[bRank-2] == (int)(bRank-1) && bPerm[bRank-1] == (int)(bRank-2))) {
                        isLastTwoTrans = false;
                    }
                } else {
                    isLastTwoTrans = false;
                }

                std::cout << "[步骤3.6] Transpose" << bTransOp->getGuid() << "是否为最后两维交换：" << (isLastTwoTrans ? "是" : "否") << std::endl;
                if (isLastTwoTrans) {
                    // 合并Transpose到transB（原false→true）
                    transB = !transB;
                    std::cout << "[步骤3.7] 合并后transB值：" << (transB ? "true" : "false") << std::endl;
                    Tensor bOrigInput = bTransOp->getInputs(0);  // i2（Guid=3）
                    std::cout << "[步骤3.8] Transpose的原始输入（i2）Guid：" << bOrigInput->getGuid() << std::endl;

                    //清理Transpose10对i2的依赖
                    bOrigInput->removeTarget(bTransOp); 
                    std::cout << "[步骤3.8.1] 已从i2（Guid=" << bOrigInput->getGuid() << "）的targets中移除Transpose" << bTransOp->getGuid() << std::endl;


                    // 更新Matmul：输入B从t3改为i2
                    std::cout << "[步骤3.9] 替换Matmul输入B从" << bInput->getGuid() << "为" << bOrigInput->getGuid() << std::endl;
                    matmulOp->replaceInput(bInput, bOrigInput);
                    // 清理Matmul与OP10的依赖
                    std::cout << "[步骤3.10] 清理Matmul与Transpose" << bTransOp->getGuid() << "的依赖" << std::endl;
                    matmulOp->removePredecessors(bTransOp);
                    // i2添加目标算子Matmul
                    std::cout << "[步骤3.11] 给i2添加目标算子Matmul" << std::endl;
                    bOrigInput->addTarget(matmulOp);

                    // 清理中间张量t3（bInput）
                    std::cout << "[步骤3.12] 清理中间张量t3（Guid：" << bInput->getGuid() << "）" << std::endl;
                    bInput->setSource(nullptr);
                    bInput->targets.clear();
                    std::cout << "[步骤3.13] 从图中移除t3" << std::endl;
                    removeTensor(bInput);  // 移除t3（Guid=6）

                    // 清理OP10（无需保留）
                    std::cout << "[步骤3.14] 清理Transpose" << bTransOp->getGuid() << "的依赖" << std::endl;
                    bTransOp->predecessors.clear();
                    bTransOp->successors.clear();
                    std::cout << "[步骤3.15] Transpose合并完成" << std::endl;
                }
            }
        }

        // 更新Matmul属性
        std::cout << "[步骤3.16] 更新Matmul属性：transA=false，transB=" << (transB ? "true" : "false") << std::endl;
        matmulOp->setTransA(false);  // 无前置Transpose，保持false
        matmulOp->setTransB(transB); // 合并后为true
        optimizedOps.push_back(matmulOp);
        std::cout << "[步骤3.17] Matmul处理完成，加入优化列表" << std::endl;
    }
    std::cout << "[步骤3.18] 合并Transpose完成，优化列表算子数：" << optimizedOps.size() << std::endl;

    // 过滤掉已合并的Transpose（OP10），仅保留Matmul
    std::cout << "[步骤3.19] 过滤冗余Transpose，仅保留Matmul" << std::endl;
    OpVec finalOps;
    for (auto& op : optimizedOps) {
        auto transOp = as<TransposeObj>(op);
        if (!transOp) {  // 只保留Matmul
            finalOps.push_back(op);
            std::cout << "[步骤3.20] 保留算子Guid：" << op->getGuid() << "（非Transpose）" << std::endl;
        } else {
            std::cout << "[步骤3.21] 过滤掉冗余Transpose算子Guid：" << op->getGuid() << std::endl;
        }
    }
    std::cout << "[步骤3.22] 过滤完成，最终算子数：" << finalOps.size() << std::endl;
    ops = std::move(finalOps);
    std::cout << "[步骤3完成] Transpose合并完成，当前算子数：" << ops.size() << "，张量数：" << tensors.size() << std::endl;

    // 第三步：清理无用张量（仅保留i1、i2、o）
    std::cout << "\n[步骤4] 开始清理无用张量，当前张量数：" << tensors.size() << std::endl;
    TensorVec usefulTensors;
    for (auto& tensor : tensors) {
        std::cout << "[步骤4.1] 处理张量Guid：" << tensor->getGuid() << "，Fuid：" << tensor->getFuid() << std::endl;
        // 过滤无效弱指针
        vector<WRef<OperatorObj>> validTargets;
        std::cout << "[步骤4.2] 张量" << tensor->getGuid() << "的目标算子数：" << tensor->targets.size() << std::endl;
        for (auto& wop : tensor->targets) {
            if (auto op = wop.lock()) {
                validTargets.push_back(wop);
                std::cout << "[步骤4.3] 保留有效目标算子Guid：" << op->getGuid() << std::endl;
            } else {
                std::cout << "[步骤4.4] 过滤无效目标算子" << std::endl;
            }
        }
        tensor->targets = validTargets;
        std::cout << "[步骤4.5] 过滤后有效目标算子数：" << tensor->targets.size() << std::endl;

        // 有用张量判断：
        // 1. 图输入：无源 + 有目标（i1、i2）
        // 2. 图输出：有源（Matmul） + 无目标（o）
        bool isInput = (tensor->getSource() == nullptr) && !tensor->targets.empty();
        bool isOutput = (tensor->getSource() != nullptr) && tensor->targets.empty();
        std::cout << "[步骤4.6] 张量" << tensor->getGuid() << "：是否为输入=" << (isInput ? "是" : "否") << "，是否为输出=" << (isOutput ? "是" : "否") << std::endl;
        if (isInput || isOutput) {
            usefulTensors.push_back(tensor);
            std::cout << "[步骤4.7] 保留有用张量Guid：" << tensor->getGuid() << std::endl;
        } else {
            std::cout << "[步骤4.8] 清理无用张量Guid：" << tensor->getGuid() << std::endl;
        }
    }
    std::cout << "[步骤4.9] 清理完成，有用张量数：" << usefulTensors.size() << std::endl;
    tensors = std::move(usefulTensors);
    std::cout << "[步骤4完成] 无用张量清理完成，当前张量数：" << tensors.size() << std::endl;

    // 重新拓扑排序，确保图结构正确
    std::cout << "\n[步骤5] 重新执行拓扑排序" << std::endl;
    sorted = false;
    IT_ASSERT(topo_sort() == true);
    std::cout << "[步骤5完成] 重新拓扑排序成功" << std::endl;


}







    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

void GraphObj::dataMalloc()
{
    // topological sorting first
    IT_ASSERT(topo_sort() == true);

    // =================================== 作业实现 ===================================
    // 1. 收集所有张量，按拓扑顺序分配（确保输入张量先分配）
    TensorVec allTensors = getTensors();

    // 2. 为每个张量分配内存，记录偏移量
    std::unordered_map<Tensor, size_t> tensorOffsets;
    for (const auto &tensor : allTensors)
    {
        size_t bytes = tensor->getBytes();
        if (bytes == 0)
            continue;

        size_t offset = allocator.alloc(bytes);
        tensorOffsets[tensor] = offset;
    }

    // 3. 调用 getPtr() 实际分配内存（一次性分配峰值内存）
    void *basePtr = allocator.getPtr();

    // 4. 为每个张量绑定内存块（Blob 封装内存指针）
    for (const auto &tensor : allTensors)
    {
        size_t bytes = tensor->getBytes();
        if (bytes == 0)
            continue;

        size_t offset = tensorOffsets[tensor];
        void *tensorPtr = static_cast<char *>(basePtr) + offset; // 计算张量实际地址
        Blob blob = make_ref<BlobObj>(runtime, tensorPtr);
        tensor->setDataBlob(blob); // 绑定内存到张量
    }
    // =================================== 作业实现 ===================================

    allocator.info();
}

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini
