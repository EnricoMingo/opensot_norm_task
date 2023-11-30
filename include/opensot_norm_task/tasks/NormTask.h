/*
 * Copyright (C) 2023 inria
 * Authors: Enrico Mingo Hoffman
 * email:  enrico.mingo-hoffman@inria.fr
 * Permission is granted to copy, distribute, and/or modify this program
 * under the terms of the GNU Lesser General Public License, version 2 or any
 * later version published by the Free Software Foundation.
 *
 * A copy of the license can be found at
 * https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details
*/

#ifndef _OPENSOT_NORM_TASK_
#define _OPENSOT_NORM_TASK_

#include <OpenSoT/Task.h>

namespace OpenSoT { namespace task {

class NormTask: public Task<Eigen::MatrixXd, Eigen::VectorXd> {

public:
    typedef std::shared_ptr<NormTask> Ptr;

    NormTask(TaskPtr taskPtr, bool marey_gain = true);
    ~NormTask(){}
    bool setRegularization(const double rho);

    /**
     * @brief setThresholds for computation of Null-Space Projector gain
     * @param e0 >= 0.
     * @param e1 >= 0.
     * @return true if e1 >= e0
     */
    bool setThresholds(double e0, double e1);

    bool setThresholdE0(double e0);

    bool setThresholdE1(double e1);

    const double getE0() const {return _e0;}
    const double getE1() const {return _e1;}


private:
    TaskPtr _taskPtr;
    virtual void _log(XBot::MatLogger2::Ptr logger) override;
    virtual void _update(const Eigen::VectorXd& x);

    /**
     * @brief computeLame Compute coefficient in equation (26)
     * @param norm_e sqrt(e'*e), norm of the error
     * @return coefficient dependent on error and thresholds
     */
    double compute_lam_e(double norm_e);

    /**
     * @brief compute_lam compute lam projector gain in equation (25)
     * @param norm_e sqrt(e'*e), norm of the error
     * @return gain dependent on error and thresholds
     */
    double compute_lam(double norm_e);

    double _norm_b;
    double _rho;
    Eigen::VectorXd _ones;
    Eigen::VectorXd _zeros;

    double _lam, _e1, _e0, _lam0, _lam1, _g;
    bool _marey_gain;
};

}
}



#endif
