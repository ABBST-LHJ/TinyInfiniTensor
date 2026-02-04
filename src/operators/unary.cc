#include "operators/unary.h"

namespace infini
{
    UnaryObj::UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output)
        : OperatorObj(type, {input}, {output})
    {
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> UnaryObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        return {{A->getDims()}};
    }

    std::string UnaryObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    ClipObj::ClipObj(GraphObj *graph, Tensor input, Tensor output,
                     std::optional<float> min, std::optional<float> max)
        : OperatorObj(OpType::Clip, {input}, {output}), minValue(min),
          maxValue(max)
    {
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> ClipObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业实现 ===================================
        IT_ASSERT(!inputs.empty(), "Clip 算子必须有一个输入张量");
        const auto inputTensor = inputs[0];
        auto outputShape = inputTensor->getDims(); // Clip 不改变张量形状
        return {{outputShape}}; // 返回与输入一致的形状（vector<Shape> 格式）
        // =================================== 作业实现 ===================================
    }

    std::string ClipObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        if (minValue.has_value())
            os << "min=" << *minValue << ",";
        if (maxValue.has_value())
            os << "max=" << *maxValue << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    CastObj::CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type)
        : OperatorObj(OpType::Cast, {input}, {output}), castType(type)
    {
        IT_ASSERT(checkValid(graph));
    }

    vector<DataType> CastObj::inferDataType(const TensorVec &inputs) const
    {
        // =================================== 作业实现 ===================================
        IT_ASSERT(!inputs.empty(), "Cast 算子必须有一个输入张量");
        // 根据 CastType 获取目标数据类型（复用已实现的 getOutputDataType 方法）
        DataType targetDtype = getOutputDataType();
        // 返回输出张量的数据类型（vector 长度=1，对应单个输出张量）
        return {targetDtype};
        // =================================== 作业实现 ===================================
    }

    optional<vector<Shape>> CastObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业实现 ===================================
        IT_ASSERT(!inputs.empty(), "Cast 算子必须有一个输入张量");
        const auto inputTensor = inputs[0];
        auto outputShape = inputTensor->getDims(); // Cast 不改变张量形状
        return {{outputShape}}; // 返回与输入一致的形状（vector<Shape> 格式）
        // =================================== 作业实现 ===================================
    }

    std::string CastObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "from=" << inputs[0]->getDType().toString() << ",";
        os << "to=" << getOutputDataType().toString() << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    DataType CastObj::getOutputDataType() const
    {
        switch (castType)
        {
        case CastType::Float2Float16:
            return DataType::Float16;
        case CastType::Float2Int64:
            return DataType::Int64;
        case CastType::Float2Int32:
            return DataType::Int32;
        case CastType::Float2Int16:
            return DataType::Int16;
        case CastType::Float2Int8:
            return DataType::Int8;
        case CastType::Int322Float:
            return DataType::Float32;
        case CastType::Int322Int8:
            return DataType::Int8;
        case CastType::Int322Int16:
            return DataType::Int16;
        case CastType::Int162Float:
            return DataType::Float32;
        case CastType::Int162Int32:
            return DataType::Int32;
        case CastType::Int82Float:
            return DataType::Float32;
        case CastType::Int82Int16:
            return DataType::Int16;
        case CastType::Int82Int32:
            return DataType::Int32;
        case CastType::Uint82Float:
            return DataType::Float32;
        case CastType::Uint82Int32:
            return DataType::Int32;
        case CastType::Uint82Int64:
            return DataType::Int64;
        case CastType::Int322Int64:
            return DataType::Int64;
        case CastType::Int642Int32:
            return DataType::Int32;
        case CastType::Int642Uint32:
            return DataType::UInt32;
        case CastType::Int642Float:
            return DataType::Float32;
        case CastType::Uint322Int64:
            return DataType::Int64;
        case CastType::Float162Float:
            return DataType::Float32;
        case CastType::BFloat162Float:
            return DataType::Float32;
        case CastType::Float2BFloat16:
            return DataType::BFloat16;
        case CastType::Float2Float:
            return DataType::Float32;
        default:
            IT_TODO_HALT();
        }
    }
}; // namespace infini
