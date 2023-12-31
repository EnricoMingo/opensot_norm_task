#include <gtest/gtest.h>
#include <OpenSoT/constraints/Aggregated.h>
#include <OpenSoT/tasks/Aggregated.h>
#include <OpenSoT/constraints/velocity/JointLimits.h>
#include <OpenSoT/constraints/velocity/VelocityLimits.h>
#include <OpenSoT/tasks/velocity/CoM.h>
#include <OpenSoT/tasks/velocity/Cartesian.h>
#include <opensot_norm_task/tasks/NormTask.h>
#include <OpenSoT/solvers/QPOasesBackEnd.h>
#include <OpenSoT/tasks/velocity/Postural.h>
#include <fstream>
#include <OpenSoT/solvers/iHQP.h>
#include <XBotInterface/ModelInterface.h>
#include <OpenSoT/utils/AutoStack.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <OpenSoT/Task.h>

namespace{

class testNormTask: public ::testing::Test
{
    protected:
    testNormTask()
    {
        std::string coman_urdf_path = ros::package::getPath("opensot_norm_task")+"/tests/panda.urdf";
        std::cout<<"coman_urdf_path: "<<coman_urdf_path<<std::endl;

        std::string coman_srdf_path = ros::package::getPath("opensot_norm_task")+"/tests/panda.srdf";
        std::cout<<"coman_srdf_path: "<<coman_srdf_path<<std::endl;

        XBot::ConfigOptions opt;
        opt.set_urdf_path(coman_urdf_path);
        opt.set_srdf_path(coman_srdf_path);
        opt.set_parameter<bool>("is_model_floating_base", false);
        opt.set_parameter<std::string>("model_type", "RBDL");
        opt.generate_jidmap();
        _model = XBot::ModelInterface::getModel(opt);

        q.setZero(_model->getJointNum());
        setHomingPosition();
        std::cout<<"q: "<<q.transpose()<<std::endl;
    }

    void setHomingPosition() {
        q[_model->getDofIndex("panda_joint1")] = 0.;
        q[_model->getDofIndex("panda_joint2")] = -0.4490246022952619;
        q[_model->getDofIndex("panda_joint3")] = 0.;
        q[_model->getDofIndex("panda_joint4")] = -2.5854036569424523;
        q[_model->getDofIndex("panda_joint5")] = 0.;
        q[_model->getDofIndex("panda_joint6")] = 2.136379054677426;
        q[_model->getDofIndex("panda_joint7")] = 0.;
    }

    virtual ~testNormTask()
    {

    }

    virtual void SetUp() {

    }

    virtual void TearDown() {

    }

public:
    XBot::ModelInterface::Ptr _model;
    Eigen::VectorXd q;

};

XBot::MatLogger2::Ptr getLogger(const std::string& name)
{
    XBot::MatLogger2::Ptr logger = XBot::MatLogger2::MakeLogger(name); // date-time automatically appended
    logger->set_buffer_mode(XBot::VariableBuffer::Mode::circular_buffer);
    return logger;
}

TEST_F(testNormTask, testAggregation)
{
    OpenSoT::tasks::velocity::Cartesian::Ptr ee1 = std::make_shared<OpenSoT::tasks::velocity::Cartesian>("ee1",this->q, *this->_model.get(), "panda_link8", "panda_link0");
    OpenSoT::tasks::velocity::Cartesian::Ptr ee2 = std::make_shared<OpenSoT::tasks::velocity::Cartesian>("ee2",this->q, *this->_model.get(), "panda_link4", "panda_link0");

    OpenSoT::task::NormTask::Ptr een1 = std::make_shared<OpenSoT::task::NormTask>(ee1, false);
    OpenSoT::task::NormTask::Ptr een2 = std::make_shared<OpenSoT::task::NormTask>(ee2, false);

    auto aggregated_task = een1 + een2;

    Eigen::MatrixXd I; I.setIdentity(2,2);

    EXPECT_TRUE(aggregated_task->getTaskID() == "norm_ee1+norm_ee2");
    EXPECT_TRUE(aggregated_task->getWeight() == I);

    std::list<OpenSoT::Task<Eigen::MatrixXd, Eigen::VectorXd>::TaskPtr> task_list;
    task_list.push_back(een1);
    task_list.push_back(een2);
    OpenSoT::tasks::Aggregated::Ptr aggr = std::make_shared<OpenSoT::tasks::Aggregated>(task_list, ee1->getXSize());
    EXPECT_TRUE(aggr->getTaskID() == "norm_ee1+norm_ee2");
    EXPECT_TRUE(aggr->getWeight() == I);


    OpenSoT::AutoStack::Ptr stack;
    stack /= (een1 + een2);

    OpenSoT::solvers::iHQP::Ptr solver = std::make_shared<OpenSoT::solvers::iHQP>(*stack, 1e10);

    OpenSoT::AutoStack::Ptr stack2 = ee1/(een1 + een2)/ee2;
    OpenSoT::solvers::iHQP::Ptr solver2 = std::make_shared<OpenSoT::solvers::iHQP>(*stack2, 1e10);

}

TEST_F(testNormTask, testRegularization)
{
    XBot::MatLogger2::Ptr logger = getLogger("testNormalTask_convergence_regularization");

    OpenSoT::tasks::velocity::Cartesian::Ptr ee = std::make_shared<OpenSoT::tasks::velocity::Cartesian>("ee",this->q, *this->_model.get(), "panda_link8", "panda_link0");
    OpenSoT::task::NormTask::Ptr een = std::make_shared<OpenSoT::task::NormTask>(ee);
    een->setLambda(.025);
    een->update(this->q);


    Eigen::Affine3d Tinit;
    ee->getActualPose(Tinit);

    Eigen::Affine3d Tgoal = Tinit;
    Tgoal.translation()[1] += 0.1;

    ee->setReference(Tgoal);

    std::cout<<"Tinit: \n"<<Tinit.matrix()<<std::endl;
    std::cout<<std::endl;
    std::cout<<"Tgoal: \n"<<Tgoal.matrix()<<std::endl;

    OpenSoT::AutoStack::Ptr stack;
    stack /= een;

    OpenSoT::solvers::iHQP::Ptr solver = std::make_shared<OpenSoT::solvers::iHQP>(*stack, 1e10);

    double rho = 1e-3;
    for(unsigned int j = 0; j < 4; ++j)
    {
        logger->add("rho_"+std::to_string(j), rho);

        een->setRegularization(rho);
        this->setHomingPosition();

        Eigen::VectorXd dq;
        dq.setZero(this->q.size());
        for(unsigned int i = 0; i < 300; ++i)
        {
            this->_model->setJointPosition(this->q);
            this->_model->update();

            stack->update(this->q);

            if(!solver->solve(dq))
                dq.setZero();
            this->q += dq;

            logger->add("q_"+std::to_string(j), this->q);
            logger->add("dq_"+std::to_string(j), dq);
            logger->add("task_error_"+std::to_string(j), ee->getError());
            logger->add("task_error_norm2_"+std::to_string(j), (ee->getError().transpose() * ee->getError())[0]);
            een->log(logger);

            std::cout<<"task_error_norm2: "<<(ee->getError().transpose() * ee->getError())[0]<<std::endl;
        }

        EXPECT_LE((ee->getError().transpose() * ee->getError())[0], 1e-4);

        rho*=1e-3;
        if(rho < 1e-9)
            rho = 0.;
    }

}

TEST_F(testNormTask, testConvergence)
{
    XBot::MatLogger2::Ptr logger = getLogger("testNormalTask_convergence_comparison_single_task");

    OpenSoT::tasks::velocity::Cartesian::Ptr ee = std::make_shared<OpenSoT::tasks::velocity::Cartesian>("ee",this->q, *this->_model.get(), "panda_link8", "panda_link0");
    ee->setLambda(.1);

    Eigen::Affine3d Tinit;
    ee->getActualPose(Tinit);

    Eigen::Affine3d Tgoal = Tinit;
    Tgoal.translation()[1] += 0.1;

    ee->setReference(Tgoal);

    std::cout<<"Tinit: \n"<<Tinit.matrix()<<std::endl;
    std::cout<<std::endl;
    std::cout<<"Tgoal: \n"<<Tgoal.matrix()<<std::endl;

    OpenSoT::AutoStack::Ptr stack;
    stack /= ee;

    OpenSoT::solvers::iHQP::Ptr solver = std::make_shared<OpenSoT::solvers::iHQP>(*stack, 1e10);

    Eigen::VectorXd dq;
    dq.setZero(this->q.size());
    for(unsigned int i = 0; i < 100; ++i)
    {
        this->_model->setJointPosition(this->q);
        this->_model->update();

        stack->update(this->q);

        if(!solver->solve(dq))
            dq.setZero();
        this->q += dq;

        logger->add("q", this->q);
        logger->add("dq", dq);
        logger->add("task_error", ee->getError());
        logger->add("task_error_norm2", (ee->getError().transpose() * ee->getError())[0]);

        std::cout<<"task_error_norm2: "<<(ee->getError().transpose() * ee->getError())[0]<<std::endl;
    }

    std::cout<<"Tgoal: \n"<<Tgoal.matrix()<<std::endl;
    std::cout<<std::endl;
    Eigen::Affine3d Tactual;
    ee->getActualPose(Tactual);
    std::cout<<"Tactual: \n"<<Tactual.matrix()<<std::endl;

    this->setHomingPosition();
    this->_model->setJointPosition(this->q);
    this->_model->update();

    stack->update(this->q);
    std::cout<<"Tinit: \n"<<Tinit.matrix()<<std::endl;
    std::cout<<std::endl;
    std::cout<<"Tgoal: \n"<<Tgoal.matrix()<<std::endl;

    ee->setLambda(1.);
    ee->update(this->q);
    OpenSoT::task::NormTask::Ptr een = std::make_shared<OpenSoT::task::NormTask>(ee);
    een->setLambda(.05);
    een->update(this->q);

    std::cout<<"een->getA: "<<een->getA()<<std::endl;
    EXPECT_TRUE(een->getA().rows() == ee->getA().rows());
    EXPECT_TRUE(een->getA().cols() == this->q.size());
    std::cout<<"een->getb: "<<een->getb()<<std::endl;
    EXPECT_TRUE(een->getb().size() == ee->getb().size());
    std::cout<<"een->getWeight: "<<een->getWeight()<<std::endl;
    EXPECT_TRUE(een->getWeight().rows() == ee->getWeight().rows());
    EXPECT_TRUE(een->getWeight().cols() == ee->getWeight().cols());



    OpenSoT::AutoStack::Ptr stack2;
    stack2 /= een;

    solver.reset();
    solver = std::make_shared<OpenSoT::solvers::iHQP>(*stack2, 1e10);

    dq.setZero(this->q.size());
    for(unsigned int i = 0; i < 100; ++i)
    {
        this->_model->setJointPosition(this->q);
        this->_model->update();

        stack2->update(this->q);

        if(!solver->solve(dq))
            dq.setZero();
        this->q += dq;

        logger->add("n_q", this->q);
        logger->add("n_dq", dq);
        logger->add("n_task_error", ee->getError());
        double norm2_b = (ee->getError().transpose() * ee->getError())[0];
        logger->add("n_task_error_norm2", norm2_b);
        logger->add("n_task_error_norm", std::sqrt(norm2_b));
        een->log(logger);

        std::cout<<"n_task_error_norm2: "<<norm2_b<<std::endl;
    }

    std::cout<<"Tgoal: \n"<<Tgoal.matrix()<<std::endl;
    std::cout<<std::endl;
    ee->getActualPose(Tactual);
    std::cout<<"Tactual: \n"<<Tactual.matrix()<<std::endl;

}

TEST_F(testNormTask, testConvergencePostural)
{
    XBot::MatLogger2::Ptr logger = getLogger("testNormalTask_convergence_comparison_task_postural");

    OpenSoT::tasks::velocity::Cartesian::Ptr ee = std::make_shared<OpenSoT::tasks::velocity::Cartesian>("ee",this->q, *this->_model.get(), "panda_link8", "panda_link0");
    ee->setLambda(.1);

    Eigen::Affine3d Tinit;
    ee->getActualPose(Tinit);

    Eigen::Affine3d Tgoal = Tinit;
    Tgoal.translation()[1] += 0.1;

    ee->setReference(Tgoal);

    std::cout<<"Tinit: \n"<<Tinit.matrix()<<std::endl;
    std::cout<<std::endl;
    std::cout<<"Tgoal: \n"<<Tgoal.matrix()<<std::endl;

    Eigen::VectorXd qmin, qmax;
    this->_model->getJointLimits(qmin, qmax);
    OpenSoT::constraints::velocity::JointLimits::Ptr jl = std::make_shared<OpenSoT::constraints::velocity::JointLimits>(this->q, qmax, qmin);

    OpenSoT::tasks::velocity::Postural::Ptr jp = std::make_shared<OpenSoT::tasks::velocity::Postural>(this->q);
    jp->setLambda(0.01);

    OpenSoT::AutoStack::Ptr stack;
    stack = (ee/jp);
    //stack<<jl;

    OpenSoT::solvers::iHQP::Ptr solver = std::make_shared<OpenSoT::solvers::iHQP>(*stack, 1e10);

    Eigen::VectorXd dq;
    dq.setZero(this->q.size());
    for(unsigned int i = 0; i < 200; ++i)
    {
        this->_model->setJointPosition(this->q);
        this->_model->update();

        stack->update(this->q);

        if(!solver->solve(dq))
            dq.setZero();
        this->q += dq;

        logger->add("q", this->q);
        logger->add("dq", dq);
        logger->add("task_error", ee->getError());
        logger->add("postural_error_norm2", (jp->getError().transpose() * jp->getError())[0]);
        logger->add("task_error_norm2", (ee->getError().transpose() * ee->getError())[0]);

        std::cout<<"task_error_norm2: "<<(ee->getError().transpose() * ee->getError())[0]<<std::endl;
    }

    std::cout<<"Tgoal: \n"<<Tgoal.matrix()<<std::endl;
    std::cout<<std::endl;
    Eigen::Affine3d Tactual;
    ee->getActualPose(Tactual);
    std::cout<<"Tactual: \n"<<Tactual.matrix()<<std::endl;

    this->setHomingPosition();
    this->_model->setJointPosition(this->q);
    this->_model->update();

    stack->update(this->q);
    std::cout<<"Tinit: \n"<<Tinit.matrix()<<std::endl;
    std::cout<<std::endl;
    std::cout<<"Tgoal: \n"<<Tgoal.matrix()<<std::endl;

    ee->setLambda(1.0);
    ee->update(this->q);
    OpenSoT::task::NormTask::Ptr een = std::make_shared<OpenSoT::task::NormTask>(ee);
    een->setLambda(.05);
    //een->setRegularization(0.);
    een->update(this->q);


    OpenSoT::AutoStack::Ptr stack2;
    stack2 = (een/jp);
    //stack2<<jl;

    solver.reset();
    solver = std::make_shared<OpenSoT::solvers::iHQP>(*stack2, 1e10);

    dq.setZero(this->q.size());
    for(unsigned int i = 0; i < 200; ++i)
    {
        this->_model->setJointPosition(this->q);
        this->_model->update();

        stack2->update(this->q);

        if(!solver->solve(dq))
            dq.setZero();
        this->q += dq;

        logger->add("n_q", this->q);
        logger->add("n_dq", dq);
        logger->add("n_task_error", ee->getError());
        logger->add("n_postural_error_norm2", (jp->getError().transpose() * jp->getError())[0]);
        double norm2_b = (ee->getError().transpose() * ee->getError())[0];
        logger->add("n_task_error_norm2", norm2_b);
        logger->add("n_task_error_norm", std::sqrt(norm2_b));
        een->log(logger);

        std::cout<<"n_task_error_norm2: "<<(ee->getError().transpose() * ee->getError())[0]<<std::endl;
    }

    std::cout<<"Tgoal: \n"<<Tgoal.matrix()<<std::endl;
    std::cout<<std::endl;
    ee->getActualPose(Tactual);
    std::cout<<"Tactual: \n"<<Tactual.matrix()<<std::endl;

}

TEST_F(testNormTask, testConvergenceCartesian)
{
    XBot::MatLogger2::Ptr logger = getLogger("testNormalTask_convergence_comparison_task_cartesian");

    OpenSoT::tasks::velocity::Cartesian::Ptr ee = std::make_shared<OpenSoT::tasks::velocity::Cartesian>("ee",this->q, *this->_model.get(), "panda_link8", "panda_link0");
    ee->setLambda(0.1);

    Eigen::Affine3d Tinit;
    ee->getActualPose(Tinit);

    Eigen::Affine3d Tgoal = Tinit;
    Tgoal.translation()[1] += 0.1;

    ee->setReference(Tgoal);

    std::cout<<"Tinit: \n"<<Tinit.matrix()<<std::endl;
    std::cout<<std::endl;
    std::cout<<"Tgoal: \n"<<Tgoal.matrix()<<std::endl;


    OpenSoT::tasks::velocity::Cartesian::Ptr el = std::make_shared<OpenSoT::tasks::velocity::Cartesian>("ee",this->q, *this->_model.get(), "panda_link4", "panda_link0");
    el->setLambda(0.1);

    Eigen::Affine3d Telinit;
    el->getActualPose(Telinit);

    Eigen::Affine3d Telgoal = Telinit;
    Telgoal.translation()[1] -= 0.1;

    el->setReference(Telgoal);

    std::cout<<"Telinit: \n"<<Telinit.matrix()<<std::endl;
    std::cout<<std::endl;
    std::cout<<"Telgoal: \n"<<Telgoal.matrix()<<std::endl;

    Eigen::VectorXd qmin, qmax;
    this->_model->getJointLimits(qmin, qmax);
    OpenSoT::constraints::velocity::JointLimits::Ptr jl = std::make_shared<OpenSoT::constraints::velocity::JointLimits>(this->q, qmax, qmin);


    OpenSoT::AutoStack::Ptr stack;
    stack = (ee/el);
    //stack<<jl;

    OpenSoT::solvers::iHQP::Ptr solver = std::make_shared<OpenSoT::solvers::iHQP>(*stack, 1e10);

    Eigen::VectorXd dq;
    dq.setZero(this->q.size());
    for(unsigned int i = 0; i < 100; ++i)
    {
        this->_model->setJointPosition(this->q);
        this->_model->update();

        stack->update(this->q);

        if(!solver->solve(dq))
            dq.setZero();
        this->q += dq;

        logger->add("q", this->q);
        logger->add("dq", dq);
        logger->add("task_error", ee->getError());
        logger->add("second_task_error", el->getError());
        logger->add("task_error_norm2", (ee->getError().transpose() * ee->getError())[0]);
        logger->add("second_task_error_norm2", (el->getError().transpose() * el->getError())[0]);

        std::cout<<"task_error_norm2: "<<(ee->getError().transpose() * ee->getError())[0]<<"     second_task_error_norm2: "<<(el->getError().transpose() * el->getError())[0]<<std::endl;
    }

    std::cout<<"Tgoal: \n"<<Tgoal.matrix()<<std::endl;
    std::cout<<std::endl;
    Eigen::Affine3d Tactual;
    ee->getActualPose(Tactual);
    std::cout<<"Tactual: \n"<<Tactual.matrix()<<std::endl;

    std::cout<<"Telgoal: \n"<<Telgoal.matrix()<<std::endl;
    std::cout<<std::endl;
    Eigen::Affine3d Telactual;
    ee->getActualPose(Telactual);
    std::cout<<"Telactual: \n"<<Telactual.matrix()<<std::endl;

    this->setHomingPosition();
    this->_model->setJointPosition(this->q);
    this->_model->update();

    stack->update(this->q);
    std::cout<<"Tinit: \n"<<Tinit.matrix()<<std::endl;
    std::cout<<std::endl;
    std::cout<<"Tgoal: \n"<<Tgoal.matrix()<<std::endl;

    ee->setLambda(1.);
    ee->update(this->q);
    OpenSoT::task::NormTask::Ptr een = std::make_shared<OpenSoT::task::NormTask>(ee);
    ee->setLambda(0.05);
    een->update(this->q);

    std::cout<<"een->getA: "<<een->getA()<<std::endl;
    if(een->getMareyGainFlag())
        EXPECT_TRUE(een->getA().rows() == 6);
    else
        EXPECT_TRUE(een->getA().rows() == 1);
    EXPECT_TRUE(een->getA().cols() == this->q.size());
    std::cout<<"een->getb: "<<een->getb()<<std::endl;
    if(een->getMareyGainFlag())
        EXPECT_TRUE(een->getb().size() == 6);
    else
        EXPECT_TRUE(een->getb().size() == 1);
    std::cout<<"een->getWeight: "<<een->getWeight()<<std::endl;
    if(een->getMareyGainFlag())
    {
        EXPECT_TRUE(een->getWeight().rows() == 6);
        EXPECT_TRUE(een->getWeight().cols() == 6);
    }
    else
    {
        EXPECT_TRUE(een->getWeight().rows() == 1);
        EXPECT_TRUE(een->getWeight().cols() == 1);
    }



    OpenSoT::AutoStack::Ptr stack2;
    stack2 = (een/el);
    //stack2<<jl;

    solver.reset();
    solver = std::make_shared<OpenSoT::solvers::iHQP>(*stack2, 1e10);

    dq.setZero(this->q.size());
    for(unsigned int i = 0; i < 100; ++i)
    {
        this->_model->setJointPosition(this->q);
        this->_model->update();

        stack2->update(this->q);

        if(!solver->solve(dq))
            dq.setZero();
        this->q += dq;

        logger->add("n_q", this->q);
        logger->add("n_dq", dq);
        logger->add("n_task_error", ee->getError());
        logger->add("n_second_task_error", el->getError());
        logger->add("n_task_error_norm2", (ee->getError().transpose() * ee->getError())[0]);
        logger->add("n_second_task_error_norm2", (el->getError().transpose() * el->getError())[0]);

        std::cout<<"task_error_norm2: "<<(ee->getError().transpose() * ee->getError())[0]<<"     second_task_error_norm2: "<<(el->getError().transpose() * el->getError())[0]<<std::endl;
    }

    std::cout<<"Tgoal: \n"<<Tgoal.matrix()<<std::endl;
    std::cout<<std::endl;
    ee->getActualPose(Tactual);
    std::cout<<"Tactual: \n"<<Tactual.matrix()<<std::endl;

    std::cout<<"Telgoal: \n"<<Telgoal.matrix()<<std::endl;
    std::cout<<std::endl;
    ee->getActualPose(Telactual);
    std::cout<<"Telactual: \n"<<Telactual.matrix()<<std::endl;

}

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
