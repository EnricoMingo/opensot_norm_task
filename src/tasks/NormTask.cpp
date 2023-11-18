#include <opensot_norm_task/tasks/NormTask.h>


OpenSoT::task::NormTask::NormTask(TaskPtr taskPtr):
    Task("norm_" + taskPtr->getTaskID(), taskPtr->getXSize()),
    _taskPtr(taskPtr)
{
    _A.setZero(1, _taskPtr->getXSize());
    _b.setZero(1);

    _W.setIdentity(1,1);
}

void OpenSoT::task::NormTask::_update(const Eigen::VectorXd &x)
{
    _taskPtr->update(x);

    _norm_b = std::sqrt((_taskPtr->getb().transpose() * _taskPtr->getb())[0]);

    _b[0] = _lambda * _norm_b;
    _A = _taskPtr->getb().transpose() * _taskPtr->getA();
    _A /= _norm_b;
}

void OpenSoT::task::NormTask::_log(XBot::MatLogger2::Ptr logger)
{
    logger->add("norm_b", _norm_b);
}
