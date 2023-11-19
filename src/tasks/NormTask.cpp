#include <opensot_norm_task/tasks/NormTask.h>


OpenSoT::task::NormTask::NormTask(TaskPtr taskPtr):
    Task("norm_" + taskPtr->getTaskID(), taskPtr->getXSize()),
    _taskPtr(taskPtr),
    _rho(1e-6)
{
    _A.setZero(1, _taskPtr->getXSize());
    _b.setZero(1);

    _W.setIdentity(1,1);

    _zeros.setZero(_taskPtr->getXSize());

    _taskPtr->update(_zeros);
    _ones.setOnes(_taskPtr->getb().size());
}

void OpenSoT::task::NormTask::_update(const Eigen::VectorXd &x)
{
    _taskPtr->update(x);

    _norm_b = std::sqrt((_taskPtr->getb().transpose() * _taskPtr->getb())[0]);

    _b[0] = _lambda * _norm_b;
    _A = (_taskPtr->getb() + _rho * _ones).transpose() * _taskPtr->getA();
    _A /= (_norm_b + _rho);
}

void OpenSoT::task::NormTask::_log(XBot::MatLogger2::Ptr logger)
{
    logger->add("norm_b", _norm_b);
}

bool OpenSoT::task::NormTask::setRegularization(const double rho)
{
    if(rho < 0.)
        return false;
    _rho = rho;
    return true;
}
