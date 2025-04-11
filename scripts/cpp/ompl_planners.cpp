#include <vector>
#include <array>
#include <utility>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <thread>

#include <Eigen/Core>


// ----------------- VAMP / Robot-Specific Includes -----------------
#include <vamp/collision/factory.hh>
#include <vamp/planning/validate.hh>
#include <vamp/robots/panda.hh>

// ----------------- OMPL Includes -----------------
#include <ompl/base/MotionValidator.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/ProjectionEvaluator.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/util/Exception.h>
#include <ompl/util/Console.h>

// Various Planners
#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/geometric/planners/prm/LazyPRM.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/geometric/planners/prm/LazyPRMstar.h>
#include <ompl/geometric/planners/prm/SPARS.h>
#include <ompl/geometric/planners/prm/SPARStwo.h>

#include <ompl/geometric/planners/est/EST.h>
#include <ompl/geometric/planners/est/BiEST.h>
#include <ompl/geometric/planners/est/ProjEST.h>

#include <ompl/geometric/planners/sbl/SBL.h>
#include <ompl/geometric/planners/sbl/pSBL.h>

#include <ompl/geometric/planners/kpiece/KPIECE1.h>
#include <ompl/geometric/planners/kpiece/BKPIECE1.h>
#include <ompl/geometric/planners/kpiece/LBKPIECE1.h>

#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/LazyRRT.h>
#include <ompl/geometric/planners/rrt/pRRT.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/RRTsharp.h>
#include <ompl/geometric/planners/rrt/RRTXstatic.h>
#include <ompl/geometric/planners/rrt/SORRTstar.h>
#include <ompl/geometric/planners/rrt/TRRT.h>
#include <ompl/geometric/planners/rrt/BiTRRT.h>
#include <ompl/geometric/planners/rrt/LBTRRT.h>
#include <ompl/geometric/planners/rrt/LazyLBTRRT.h>
#include <ompl/geometric/planners/rrt/STRRTstar.h>

#include <ompl/geometric/planners/rlrt/RLRT.h>
#include <ompl/geometric/planners/rlrt/BiRLRT.h>

#include <ompl/geometric/planners/stride/STRIDE.h>
#include <ompl/geometric/planners/pdst/PDST.h>
#include <ompl/geometric/planners/fmt/FMT.h>
#include <ompl/geometric/planners/fmt/BFMT.h>

#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/geometric/planners/informedtrees/ABITstar.h>
#include <ompl/geometric/planners/informedtrees/AITstar.h>
#include <ompl/geometric/planners/informedtrees/EITstar.h>
#include <ompl/geometric/planners/informedtrees/EIRMstar.h>

// ----------------- Include the SDF-to-PointCloud header -----------------
#include "sdf_to_pointcloud.hpp"
#include <algorithm>

// Namespaces
namespace ob = ompl::base;
namespace og = ompl::geometric;

// CustomProjectionEvaluator for 7-DOF manipulator planning
class CustomProjectionEvaluator : public ob::ProjectionEvaluator
{
public:
    CustomProjectionEvaluator(const ob::StateSpace *space) : ob::ProjectionEvaluator(space) {}

    // Return the dimension of the projection (using a 3D projection)
    unsigned int getDimension() const override { return 7; }

    // Set default cell sizes for the projection's discretization
    void defaultCellSizes() override {
        cellSizes_.resize(7);
        cellSizes_[0] = 0.3;  
        cellSizes_[1] = 0.3;
        cellSizes_[2] = 0.3;
        cellSizes_[3] = 0.3;
        cellSizes_[4] = 0.3;
        cellSizes_[5] = 0.3;
        cellSizes_[6] = 0.3;
    }

    // Compute the projection using Eigen::Ref<Eigen::VectorXd>
    void project(const ob::State *state, Eigen::Ref<Eigen::VectorXd> projection) const override {
        const auto *rstate = state->as<ob::RealVectorStateSpace::StateType>();
        
        projection[0] = rstate->values[0];  // Joint 1
        projection[1] = rstate->values[1];  // Joint 2
        projection[2] = rstate->values[2];  // Joint 3
        projection[3] = rstate->values[3];  // Joint 4
        projection[4] = rstate->values[4];  // Joint 5
        projection[5] = rstate->values[5];  // Joint 6
        projection[6] = rstate->values[6];  // Joint 7
    }
};

//Create a structure to store timing results
struct PlannerResult {
    std::string name;
    double time_ms;
    bool success;
};

// Global vector to store results across all planner runs
std::vector<PlannerResult> plannerResults;


// ----------------- Robot / VAMP Definitions -----------------
using Robot = vamp::robots::Panda;
static constexpr std::size_t dimension = Robot::dimension;
using Configuration                  = Robot::Configuration;
static constexpr std::size_t rake    = vamp::FloatVectorWidth;

using EnvironmentInput  = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;

// ----------------- Start / Goal / Obstacles -----------------
static constexpr Robot::ConfigurationArray start = {0., -0.785, 0., -2.356, 0., 1.571, 0.785};
static constexpr Robot::ConfigurationArray goal = {2.35, 1., 0., -0.8, 0, 2.5, 0.785};

static const std::vector<std::array<float, 3>> problem = {
    {0.55, 0, 0.25},
    {0.35, 0.35, 0.25},
    {0, 0.55, 0.25},
    {-0.55, 0, 0.25},
    {-0.35, -0.35, 0.25},
    {0, -0.55, 0.25},
    {0.35, -0.35, 0.25},
    {0.35, 0.35, 0.8},
    {0, 0.55, 0.8},
    {-0.35, 0.35, 0.8},
    {-0.55, 0, 0.8},
    {-0.35, -0.35, 0.8},
    {0, -0.55, 0.8},
    {0.35, -0.35, 0.8},
};

static constexpr float radius               = 0.2;
static constexpr float planning_time        = 1.0;  // each planner gets up to 1s
static constexpr float simplification_time  = 1.0;

// ----------------- Interpolate and Save Path -----------------
void saveInterpolatedPathAsCSV(const og::PathGeometric &path,
                               const ob::SpaceInformationPtr &si,
                               const std::string &filename,
                               unsigned int stepsPerSegment = 10)
{
    std::ofstream outFile("solution/" + filename);
    if (!outFile.is_open())
    {
        std::cerr << "[ERROR] Could not open file " << filename << " for writing.\n";
        return;
    }

    outFile << std::fixed << std::setprecision(6);
    outFile << "joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,joint_7\n";

    const ob::StateSpacePtr &space = si->getStateSpace();
    ob::State *temp = si->allocState();

    for (std::size_t i = 0; i + 1 < path.getStateCount(); ++i)
    {
        const ob::State *s1 = path.getState(i);
        const ob::State *s2 = path.getState(i + 1);
        for (unsigned int step = 0; step < stepsPerSegment; ++step)
        {
            double t = static_cast<double>(step) / stepsPerSegment;
            space->interpolate(s1, s2, t, temp);
            auto *rvs = temp->as<ob::RealVectorStateSpace::StateType>();
            for (std::size_t d = 0; d < dimension; ++d)
            {
                outFile << rvs->values[d];
                if (d + 1 < dimension)
                    outFile << ",";
            }
            outFile << "\n";
        }
    }

    const ob::State *last = path.getState(path.getStateCount() - 1);
    auto *rvsLast = last->as<ob::RealVectorStateSpace::StateType>();
    for (std::size_t d = 0; d < dimension; ++d)
    {
        outFile << rvsLast->values[d];
        if (d + 1 < dimension)  
            outFile << ",";
    }
    outFile << "\n";

    si->freeState(temp);
    outFile.close();
    std::cout << "[INFO] Saved interpolated path to " << filename << "\n";
}

// ----------------- Convert OMPL <-> VAMP -----------------
inline static Configuration ompl_to_vamp(const ob::State *state)
{
    alignas(Configuration::S::Alignment)
        std::array<typename Configuration::S::ScalarT, Configuration::num_scalars> aligned_buffer;

    auto *as = state->as<ob::RealVectorStateSpace::StateType>();
    for (std::size_t i = 0; i < dimension; ++i)
    {
        aligned_buffer[i] = static_cast<float>(as->values[i]);
    }
    return Configuration(aligned_buffer.data());
}

inline static void vamp_to_ompl(const Configuration &c, ob::State *state)
{
    auto *as = state->as<ob::RealVectorStateSpace::StateType>();
    for (std::size_t i = 0; i < dimension; ++i)
    {
        as->values[i] = static_cast<double>(c[{i, 0}]);
    }
}

// ----------------- StateValidityChecker -----------------
struct VAMPStateValidator : public ob::StateValidityChecker
{
    VAMPStateValidator(const ob::SpaceInformationPtr &si, const EnvironmentVector &env_v)
      : ob::StateValidityChecker(si), env_v(env_v)
    {
    }

    bool isValid(const ob::State *state) const override
    {
        auto configuration = ompl_to_vamp(state);
        return vamp::planning::validate_motion<Robot, rake, 1>(configuration, configuration, env_v);
    }

    const EnvironmentVector &env_v;
};

// ----------------- MotionValidator -----------------
struct VAMPMotionValidator : public ob::MotionValidator
{
    VAMPMotionValidator(const ob::SpaceInformationPtr &si, const EnvironmentVector &env_v)
      : ob::MotionValidator(si.get()), env_v(env_v)
    {
    }

    bool checkMotion(const ob::State *s1, const ob::State *s2) const override
    {
        return vamp::planning::validate_motion<Robot, rake, Robot::resolution>(
            ompl_to_vamp(s1), ompl_to_vamp(s2), env_v);
    }

    // auto
    // checkMotion(const ob::State *, const ob::State *, std::pair<ob::State *, double> &) const -> bool override
    // {
    //     throw ompl::Exception("Not implemented!");
    // }
    

    bool checkMotion(const ob::State *s1,
                     const ob::State *s2,
                     std::pair<ob::State *, double> &lastValid) const override
    {
        unsigned int nd = si_->getStateSpace()->validSegmentCount(s1, s2);
        if (nd <= 1)
        {
            if (checkMotion(s1, s2))
            {
                if (lastValid.first)
                    si_->copyState(lastValid.first, s2);
                lastValid.second = 1.0;
                return true;
            }
            else
            {
                if (lastValid.first)
                    si_->copyState(lastValid.first, s1);
                lastValid.second = 0.0;
                return false;
            }
        }
        else
        {
            ob::State *temp = si_->allocState();
            for (unsigned int i = 1; i <= nd; ++i)
            {
                double fraction = static_cast<double>(i) / nd;
                si_->getStateSpace()->interpolate(s1, s2, fraction, temp);
                if (!si_->isValid(temp))
                {
                    double partial = static_cast<double>(i - 1) / nd;
                    if (lastValid.first)
                        si_->getStateSpace()->interpolate(s1, s2, partial, lastValid.first);
                    lastValid.second = partial;
                    si_->freeState(temp);
                    return false;
                }
            }
            if (lastValid.first)
                si_->copyState(lastValid.first, s2);
            lastValid.second = 1.0;
            si_->freeState(temp);
            return true;
        }
    }

    const EnvironmentVector &env_v;
};

// ----------------- createPlanner -----------------
ob::PlannerPtr createPlanner(const ob::SpaceInformationPtr &si,
                             const std::string &plannerName) {
    using namespace ompl::geometric;
    // Special handling for projection-based planners
    if (plannerName == "ProjEST") {
        std::cout << "[DEBUG] Creating ProjEST planner with custom projection evaluator.\n";
        auto est = std::make_shared<ProjEST>(si);
        est->setProjectionEvaluator(std::make_shared<CustomProjectionEvaluator>(si->getStateSpace().get()));
        est->setGoalBias(0.1);      // Increased goal bias to help reach the goal faster
        est->setRange(1.0);         // Larger range for better exploration
        return est;
    } 
    else if (plannerName == "KPIECE") {
        auto kpiece = std::make_shared<KPIECE1>(si);
        kpiece->setProjectionEvaluator(si->getStateSpace()->getDefaultProjection());
        kpiece->setRange(0.3);
        kpiece->setBorderFraction(0.8);
        kpiece->setMinValidPathFraction(0.1);
        return kpiece;
    }
    else if (plannerName == "BKPIECE") {
        auto bkpiece = std::make_shared<BKPIECE1>(si);
        bkpiece->setProjectionEvaluator(si->getStateSpace()->getDefaultProjection());
        bkpiece->setRange(0.3);
        bkpiece->setBorderFraction(0.8);
        return bkpiece;
    }
    else if (plannerName == "LBKPIECE") {
        auto lbkpiece = std::make_shared<LBKPIECE1>(si);
        lbkpiece->setProjectionEvaluator(si->getStateSpace()->getDefaultProjection());
        lbkpiece->setRange(0.3);
        lbkpiece->setBorderFraction(0.8);
        return lbkpiece;
    }
    else if (plannerName == "SBL") {
        auto sbl = std::make_shared<SBL>(si);
        sbl->setProjectionEvaluator(si->getStateSpace()->getDefaultProjection());
        sbl->setRange(0.3);
        return sbl;
    }
    else if (plannerName == "pSBL") {
        auto psbl = std::make_shared<pSBL>(si);
        psbl->setProjectionEvaluator(si->getStateSpace()->getDefaultProjection());
        psbl->setRange(0.3);
        return psbl;
    }
    else if (plannerName == "PRM")                return std::make_shared<PRM>(si);
    else if (plannerName == "LazyPRM")       return std::make_shared<LazyPRM>(si);
    else if (plannerName == "PRMstar")       return std::make_shared<PRMstar>(si);
    else if (plannerName == "LazyPRMstar")   return std::make_shared<LazyPRMstar>(si);
    else if (plannerName == "SPARS")         return std::make_shared<SPARS>(si);
    else if (plannerName == "SPARS2")        return std::make_shared<SPARStwo>(si);
    else if (plannerName == "EST")           return std::make_shared<EST>(si);
    else if (plannerName == "BiEST")         return std::make_shared<BiEST>(si);
    else if (plannerName == "RRT")           return std::make_shared<RRT>(si);
    else if (plannerName == "RRTConnect")    return std::make_shared<RRTConnect>(si);
    else if (plannerName == "LazyRRT")       return std::make_shared<LazyRRT>(si);
    else if (plannerName == "pRRT")          return std::make_shared<pRRT>(si);
    else if (plannerName == "RLRT")          return std::make_shared<RLRT>(si);
    else if (plannerName == "BiRLRT")        return std::make_shared<BiRLRT>(si);
    else if (plannerName == "RRTstar")       return std::make_shared<RRTstar>(si);
    else if (plannerName == "RRTsharp")      return std::make_shared<RRTsharp>(si);
    else if (plannerName == "RRTX")          return std::make_shared<RRTXstatic>(si);
    else if (plannerName == "InformedRRTstar") return std::make_shared<InformedRRTstar>(si);
    else if (plannerName == "SORRTstar")     return std::make_shared<SORRTstar>(si);
    else if (plannerName == "BITstar")       return std::make_shared<BITstar>(si);
    else if (plannerName == "ABITstar")      return std::make_shared<ABITstar>(si);
    else if (plannerName == "AITstar")       return std::make_shared<AITstar>(si);
    else if (plannerName == "EITstar")       return std::make_shared<EITstar>(si);
    else if (plannerName == "EIRMstar")      return std::make_shared<EIRMstar>(si);
    else if (plannerName == "LBTRRT")        return std::make_shared<LBTRRT>(si);
    else if (plannerName == "LazyLBTRRT")    return std::make_shared<LazyLBTRRT>(si);
    else if (plannerName == "STRRTstar")     return std::make_shared<STRRTstar>(si);
    else if (plannerName == "TRRT")          return std::make_shared<TRRT>(si);
    else if (plannerName == "BiTRRT")        return std::make_shared<BiTRRT>(si);
    else if (plannerName == "STRIDE")        return std::make_shared<STRIDE>(si);
    else if (plannerName == "PDST")          return std::make_shared<PDST>(si);
    else if (plannerName == "FMTstar")       return std::make_shared<FMT>(si);
    else if (plannerName == "BFMTstar")      return std::make_shared<BFMT>(si);
    else {
        OMPL_ERROR("Requested planner '%s' is not recognized or not supported.", plannerName.c_str());
        return ob::PlannerPtr();
    }
}

// ----------------- runPlanner -----------------
void runPlanner(const std::string &plannerName,
                const ob::SpaceInformationPtr &si,
                const ob::ProblemDefinitionPtr &pdef,
                bool optimize)
{   
    std::cout << "\n=== [runPlanner] Running Planner: " << plannerName << " ===" << std::endl;
    if (!optimize)
        std::cout << "[DEBUG] Not optimizing -> stop at first feasible solution.\n";
    else
        std::cout << "[DEBUG] Optimizing -> might refine solutions.\n";

    ob::PlannerPtr planner = createPlanner(si, plannerName);
    planner->setProblemDefinition(pdef);
    planner->setup();

    if ( plannerName == "AITstar" || plannerName == "EITstar" || plannerName == "EIRMstar" || plannerName == "LazyLBTRRT" || 
        plannerName == "BITstar" || plannerName == "ABITstar") 
    {
        // Lowering the range from the default (~3.14) to 1.0
        // planner->params().setParam("range", "1.5");
        // std::cout << "[DEBUG] Set ProjEST range parameter to 1.5\n";
        return;
    }

    // Configure specific parameters for projection-based planners
    if (plannerName == "ProjEST" || plannerName == "KPIECE" || plannerName == "BKPIECE" || 
        plannerName == "LBKPIECE" || plannerName == "SBL" || plannerName == "pSBL") {

        //planner->params().setParam("range", "0.5");
        return;
        
    }

    

    std::cout << "[DEBUG] Planner settings:\n";
    planner->printProperties(std::cout);
    planner->printSettings(std::cout);

    ob::PlannerStatus solved;
    double elapsed_time_ms = 0.0;
    bool success = false;
    try
    {
        std::cout << "[DEBUG] Calling solve(" << planning_time << ")...\n";
        auto start_time = std::chrono::high_resolution_clock::now();

        solved = planner->solve(planning_time);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        elapsed_time_ms = elapsed.count();
        
        // Check if it was successful
        success = (solved == ob::PlannerStatus::EXACT_SOLUTION);
    }
    catch (ompl::Exception &ex)
    {
        std::cout << "[EXCEPTION] Planner " << plannerName << " threw:\n"
                  << ex.what() << "\nSkipping this planner.\n";
        planner->clear();
        pdef->clearSolutionPaths();

        plannerResults.push_back({plannerName, elapsed_time_ms, false});

        return;
    }
    plannerResults.push_back({plannerName, elapsed_time_ms, success});


    if (solved == ob::PlannerStatus::EXACT_SOLUTION)
    {
        std::cout << "  [SUCCESS] " << plannerName
                  << " found EXACT_SOLUTION in " << elapsed_time_ms << " ms.\n";
        auto path = pdef->getSolutionPath();
        auto &pathGeo = static_cast<og::PathGeometric &>(*path);

        auto obj = pdef->getOptimizationObjective();
        auto initialCost = pathGeo.cost(obj);
        std::cout << "[DEBUG] Path has " << pathGeo.getStateCount()
                  << " states before simplification.\n";

        pathGeo.print(std::cout);

        std::string filename = plannerName + std::string(".csv");
        saveInterpolatedPathAsCSV(pathGeo, si, filename, /*stepsPerSegment=*/20);
    }
    else
    {
        std::cout << "  [FAILURE] " << plannerName
                  << " did not find an EXACT_SOLUTION.\n"
                  << "  (PlannerStatus = " << solved.asString() << ")\n";
    }

    pdef->clearSolutionPaths();
    planner->clear();
    std::cout << "[DEBUG] Done with " << plannerName << "\n";
}

// ----------------- main() -----------------
int main(int argc, char **argv)
{
    ompl::msg::setLogLevel(ompl::msg::LOG_INFO);

    // Create an EnvironmentInput which will later be converted into an EnvironmentVector.
    EnvironmentInput environment;
    // Check for an optional "optimize" flag if additional arguments are given.
    bool optimize = false;
    if (argc > 2)
        optimize = true;

    if (argc > 1)
    {
        // --- SDF mode ---
        // When SDF file is provided, process the file and build the environment using CAPT.

        std::string sdfFilename = argv[1];

        // Process the SDF file using SDFToPointCloud.
        SDFToPointCloud sdfConverter(std::thread::hardware_concurrency(), 0.01);
        if (!sdfConverter.loadSDF(sdfFilename))
        {
            std::cerr << "Failed to load SDF file: " << sdfFilename << std::endl;
            return 1;
        }

        // Obtain the generated point cloud.
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = sdfConverter.getPointCloud();
        std::cout << "[DEBUG] SDF loaded. Generated point cloud with " 
                  << cloud->points.size() << " points.\n";

        // Convert the pcl point cloud into a vector of 3D points.
        std::vector<std::array<float, 3>> captPoints;
        captPoints.reserve(cloud->points.size());
        for (const auto &pt : cloud->points)
        {
            captPoints.push_back({pt.x, pt.y, pt.z});
        }

        // Build the environment using CAPT. Set an appropriate radius for each point.
        float pointRadius = 0.01f;
        environment.pointclouds.push_back(vamp::collision::CAPT(captPoints, pointRadius, 0.01f, 0.08f));

    }
    else
    {
        // --- Default problem mode ---
        // No SDF file is provided; fall back to using a predefined problem (example spheres).

        std::cout << "[DEBUG] No SDF file provided. Building default environment using problem configuration...\n";

        // Example: Define a simple problem with sphere centers.
        std::vector<std::array<float, 3>> problem = {
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f}
        };

        // Set a radius for the spheres.
        float radius = 0.05f;
        // Use the factory function to create spheres from the problem data.
        for (const auto &sphere_center : problem)
        {
            environment.spheres.push_back(vamp::collision::factory::sphere::array(sphere_center, radius));
        }
    }

    // Convert the EnvironmentInput into an EnvironmentVector.
    auto env_v = EnvironmentVector(environment);

    // Create RealVectorStateSpace and set bounds
    auto space = std::make_shared<ob::RealVectorStateSpace>(dimension);
    static constexpr std::array<float, dimension> zeros = {0., 0., 0., 0., 0., 0., 0.};
    static constexpr std::array<float, dimension> ones  = {1., 1., 1., 1., 1., 1., 1.};
    Configuration zero_v(zeros), one_v(ones);

    Robot::scale_configuration(zero_v);
    Robot::scale_configuration(one_v);

    ob::RealVectorBounds bounds(dimension);
    for (std::size_t i = 0; i < dimension; ++i)
    {
        bounds.setLow(i, zero_v[{i, 0}]);
        bounds.setHigh(i, one_v[{i, 0}]);
    }
    space->setBounds(bounds);

    space->registerDefaultProjection(std::make_shared<CustomProjectionEvaluator>(space.get()));

    space->setLongestValidSegmentFraction(0.01);

    // SpaceInformation
    auto si = std::make_shared<ob::SpaceInformation>(space);
    si->setStateValidityChecker(std::make_shared<VAMPStateValidator>(si, env_v));
    si->setMotionValidator(std::make_shared<VAMPMotionValidator>(si, env_v));
    si->setup();

    // Start/Goal
    ob::ScopedState<> start_ompl(space), goal_ompl(space);
    for (std::size_t i = 0; i < dimension; ++i)
    {
        start_ompl[i] = start[i];
        goal_ompl[i]  = goal[i];
    }
    std::cout << "[DEBUG main()] Start/Goal set.\n";

    // ProblemDefinition
    auto pdef = std::make_shared<ob::ProblemDefinition>(si);
    pdef->setStartAndGoalStates(start_ompl, goal_ompl);

    auto obj = std::make_shared<ob::PathLengthOptimizationObjective>(si);
    pdef->setOptimizationObjective(obj);

    if (!optimize)
        obj->setCostThreshold(ob::Cost(std::numeric_limits<double>::infinity()));
    
    // Reset the results vector
    plannerResults.clear();

    // List of all supported planner names
    std::vector<std::string> allPlanners = {
        "PRM", "LazyPRM", "PRMstar", "LazyPRMstar", "SPARS", "SPARS2",
        "EST", "BiEST", 
        "ProjEST", 
        "SBL", "pSBL",
        "KPIECE", "BKPIECE", "LBKPIECE",
        "RRT", "RRTConnect", "LazyRRT", "pRRT", "RLRT", "BiRLRT",
        "RRTstar", "RRTsharp", "RRTX", "InformedRRTstar", "SORRTstar",
        "BITstar",
         "ABITstar", 
        "AITstar",
          "EITstar", "EIRMstar",
        "LBTRRT", "LazyLBTRRT", 
        // "STRRTstar", 
        "TRRT", "BiTRRT",
        "STRIDE", "PDST", "FMTstar", "BFMTstar"
    };

    // Run each planner
    for (const auto &plannerName : allPlanners)
    {
        runPlanner(plannerName, si, pdef, optimize);
        std::cout << "-------------------------------------------\n";
    }

    std::cout << "[DEBUG main()] All planners done!\n";

     // Print out timing results
     std::cout << "\n\n=== PLANNER TIMING RESULTS ===\n";
     std::cout << std::setw(20) << "Planner Name" << std::setw(15) << "Time (ms)" << std::setw(10) << "Success" << std::endl;
     std::cout << std::string(45, '-') << std::endl;
     
     for (const auto& result : plannerResults)
     {
         std::cout << std::fixed << std::setprecision(2);
         std::cout << std::setw(20) << result.name 
                   << std::setw(15) << result.time_ms 
                   << std::setw(10) << (result.success ? "Yes" : "No") 
                   << std::endl;
     }
     
     // Find and print the fastest successful planner
     auto fastestSuccessful = std::find_if(plannerResults.begin(), plannerResults.end(), 
                                          [](const PlannerResult& r) { return r.success; });
     
     if (fastestSuccessful != plannerResults.end()) {
         for (auto it = plannerResults.begin(); it != plannerResults.end(); ++it) {
             if (it->success && it->time_ms < fastestSuccessful->time_ms) {
                 fastestSuccessful = it;
             }
         }
         
         std::cout << "\nFastest successful planner: " << fastestSuccessful->name 
                   << " (" << fastestSuccessful->time_ms << " ms)" << std::endl;
     } else {
         std::cout << "\nNo successful planners found." << std::endl;
     }

    return 0;
}
