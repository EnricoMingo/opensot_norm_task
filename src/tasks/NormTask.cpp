#include <opensot_norm_task/tasks/NormTask.h>


OpenSoT::task::NormTask::NormTask(TaskPtr taskPtr, bool marey_gain):
    Task("norm_" + taskPtr->getTaskID(), taskPtr->getXSize()),
    _taskPtr(taskPtr),
    _rho(1e-9),
    _e1(1e-1),
    _e0(1e-2),
    _lam0(1e-9),
    _lam1(0.999),
    _marey_gain(marey_gain)
{
    _zeros.setZero(_taskPtr->getXSize());
    _taskPtr->update(_zeros);

    if(_marey_gain)
    {
        _A.setZero(_taskPtr->getA().rows(), _taskPtr->getA().cols());
        _b.setZero(_taskPtr->getb().size());
        _W.setIdentity(_taskPtr->getWeight().rows(), _taskPtr->getWeight().cols());
    }
    else
    {
        _A.setZero(1, _taskPtr->getA().cols());
        _b.setZero(1);
        _W.setIdentity(1, 1);
    }

    _ones.setOnes(_taskPtr->getb().size());
}

void OpenSoT::task::NormTask::_update(const Eigen::VectorXd &x)
{
    _taskPtr->update(x);
    _norm_b = std::sqrt((_taskPtr->getb().transpose() * _taskPtr->getb())[0]);

    _b.setZero();
    _A.setZero();

    _b[0] = _lambda * _norm_b;
    _A.block(0,0,1,_A.cols()) = (_taskPtr->getb() + _rho * _ones).transpose() * _taskPtr->getA();
    _A.block(0,0,1,_A.cols()) /= (_norm_b + _rho);

    if(_marey_gain)
    {
        _g = compute_lam(_norm_b);

        _b[0] *= _g;
        _A.block(0,0,1,_A.cols()) *= _g;

        _b += (1. - _g) * _taskPtr->getb();
        _A += (1. - _g) * _taskPtr->getA();
    }
}

void OpenSoT::task::NormTask::_log(XBot::MatLogger2::Ptr logger)
{
    logger->add("norm_b", _norm_b);
    if(_marey_gain)
    {
        logger->add("g", _g);
    }
}

bool OpenSoT::task::NormTask::setRegularization(const double rho)
{
    if(rho < 0.)
        return false;
    _rho = rho;
    return true;
}

double OpenSoT::task::NormTask::compute_lam(double norm_e)
{
    if(norm_e >= _e1)
        return 1.;
    if(norm_e >= _e0)
        return (compute_lam_e(norm_e) - _lam0)/(_lam1 - _lam0);

    return 0.;
}

double OpenSoT::task::NormTask::compute_lam_e(double norm_e)
{
    double a = norm_e - _e0;
    double b =_e1 - _e0;

    double c = -12. * a/b + 6.;

    return 1./std::exp(c);
}

bool OpenSoT::task::NormTask::setThresholds(double e0, double e1)
{
    if(e1 >= e0)
    {
        _e0 = e0;
        _e1 = e1;
        return true;
    }
    return false;
}
