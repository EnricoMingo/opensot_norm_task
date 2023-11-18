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

    NormTask(TaskPtr taskPtr);
    ~NormTask(){}


private:
    TaskPtr _taskPtr;
    virtual void _log(XBot::MatLogger2::Ptr logger);
    virtual void _update(const Eigen::VectorXd& x);

    double _norm_b;
};

}
}



#endif
