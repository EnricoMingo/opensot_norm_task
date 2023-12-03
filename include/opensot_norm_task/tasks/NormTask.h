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
/**
 * @brief The NormTask class takes inspiration from the paper "A new large projection operator for the redundancy framework",
 * by  Mohammed Marey and Francois Chaumette. It implements the following method applied to a generic task ||Ax - b||:
 *
 *  A_n = b'A/||b||
 *  b_n = ||b||
 *
 * notice that given A an (m x n) task matrix, A_ will be (1 x n) since it is projected into the vector b'/||b||.
 * To avoid 0/0 when b->0, we apply the following regularization:
 *
 *  A_n = (b + rho*i)'A / (||b|| + rho)
 *  b_n = ||b||
 *
 * with i a vector of ones.
 */
class NormTask: public Task<Eigen::MatrixXd, Eigen::VectorXd> {

public:
    typedef std::shared_ptr<NormTask> Ptr;
    /**
     * @brief NormTask constructor
     * @param taskPtr task used to compute the projection
     * @param marey_gain implements the blending proposed in "A new large projection operator for the redundancy framework".
     * @note if marey_gain = false, the task will be implemented as a single row task.
     * If marey_gain = true, to implement the blending the task will be implemented as a 6 row task:
     *            _         _
     *           | b'A/||b|| |
     *           |     0     |
     *           |     0     |
     * A_ = alpha|     0     | + (1 - alpha)[A]
     *           |     0     |
     *           |_    0    _|
     *
     * and b_ = alpha[||b|| 0 0 0 0 0]' + (1-alpha)b
     *
     * with alpha computed as suggested in the paper.
     */
    NormTask(TaskPtr taskPtr, bool marey_gain = true);

    ~NormTask(){}

    /**
     * @brief setRegularization set the rho (dafault 1e-9) value used to regularize the norm
     * @param rho
     * @return false if < 0.
     */
    bool setRegularization(const double rho);

    /**
     * @brief setThresholds for computation of Null-Space Projector gain
     * @param e0 >= 0.
     * @param e1 >= 0.
     * @return true if e1 >= e0
     */
    bool setThresholds(double e0, double e1);

    /**
     * @brief setThresholdE0
     * @param e0 single parameter
     * @return true if e1 >= e0
     */
    bool setThresholdE0(double e0);

    /**
     * @brief setThresholdE1
     * @param e1 single parameter
     * @return true if e1 >= e0
     */
    bool setThresholdE1(double e1);

    /**
     * @brief getE0
     * @return threshold e0
     */
    const double getE0() const { return _e0; }

    /**
     * @brief getE1
     * @return threshold e1
     */
    const double getE1() const  { return _e1; }

    /**
     * @brief getMareyGainFlag
     * @return value of marey_gain flag
     */
    const bool getMareyGainFlag() const { return _marey_gain; }


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
