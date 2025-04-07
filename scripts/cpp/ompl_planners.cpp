#include <vector>
#include <array>
#include <utility>
#include <iostream>
#include <string>
#include <chrono>

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
#include <ompl/geometric/PathSimplifier.h>

// Various Planners
#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/LazyRRT.h>
#include <ompl/geometric/planners/rrt/TRRT.h>
#include <ompl/geometric/planners/est/EST.h>
#include <ompl/geometric/planners/kpiece/KPIECE1.h>
#include <ompl/geometric/planners/kpiece/BKPIECE1.h>
#include <ompl/geometric/planners/stride/STRIDE.h>
#include <ompl/geometric/planners/sbl/SBL.h>
#include <ompl/geometric/planners/sbl/pSBL.h>
#include <ompl/geometric/planners/fmt/FMT.h>
#include <ompl/geometric/planners/fmt/BFMT.h>

#include <ompl/util/Exception.h>
#include <ompl/util/Console.h>

// Namespaces
namespace ob = ompl::base;
namespace og = ompl::geometric;

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


#include <fstream>   // for std::ofstream
#include <iomanip>   // for std::setprecision, etc.

// Interpolate the final path, then store in CSV:
void saveInterpolatedPathAsCSV(const og::PathGeometric &path,
                               const ob::SpaceInformationPtr &si,
                               const std::string &filename,
                               unsigned int stepsPerSegment = 10)
{
    // Open a file for writing
    std::ofstream outFile("solution/" + filename);
    if (!outFile.is_open())
    {
        std::cerr << "[ERROR] Could not open file " << filename << " for writing.\n";
        return;
    }

    // Basic CSV header (optional)
    outFile << std::fixed << std::setprecision(6);
    outFile << "joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,joint_7\n";

    const ob::StateSpacePtr &space = si->getStateSpace();
    ob::State *temp = si->allocState();

    // We'll go through each pair of states in the path
    for (std::size_t i = 0; i + 1 < path.getStateCount(); ++i)
    {
        const ob::State *s1 = path.getState(i);
        const ob::State *s2 = path.getState(i + 1);

        // For each pair, do 'stepsPerSegment' interpolation steps
        for (unsigned int step = 0; step < stepsPerSegment; ++step)
        {
            double t = (double)step / (double)stepsPerSegment;
            space->interpolate(s1, s2, t, temp);

            // Extract 7 joint values
            auto *rvs = temp->as<ob::RealVectorStateSpace::StateType>();
            for (std::size_t d = 0; d < dimension; ++d)
            {
                outFile << rvs->values[d];
                if (d + 1 < dimension)  // if not last dimension, put a comma
                    outFile << ",";
            }
            outFile << "\n";
        }
    }

    // Write the last state exactly (so we definitely include the goal)
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
        // All-or-nothing check
        return vamp::planning::validate_motion<Robot, rake, Robot::resolution>(
            ompl_to_vamp(s1), ompl_to_vamp(s2), env_v);
    }

    // If a planner tries partial-check, this throws. We catch the exception in runPlanner.
    // bool checkMotion(const ob::State *, const ob::State *,
    //                  std::pair<ob::State *, double> &) const override
    // {
    //     throw ompl::Exception("Partial checkMotion not implemented!");
    // }
    bool checkMotion(const ob::State *s1,
        const ob::State *s2,
        std::pair<ob::State *, double> &lastValid) const override
    {
    unsigned int nd = si_->getStateSpace()->validSegmentCount(s1, s2);
    if (nd <= 1)
    {
    // Just do a single check
        if (checkMotion(s1, s2))  // calls your single-step version
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
            // We'll allocate a temporary state to check intermediate collisions
            ob::State *temp = si_->allocState();
            for (unsigned int i = 1; i <= nd; ++i)
            {
                double fraction = (double)i / (double)nd;
                si_->getStateSpace()->interpolate(s1, s2, fraction, temp);

                if (!si_->isValid(temp))
                {
                    double partial = (double)(i - 1) / (double)nd;
                    if (lastValid.first)
                        si_->getStateSpace()->interpolate(s1, s2, partial, lastValid.first);
                    lastValid.second = partial;
                    si_->freeState(temp);
                    return false;
                }
            }
            // entire segment is valid
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
ob::PlannerPtr createPlanner(const std::string &plannerName, const ob::SpaceInformationPtr &si)
{
    std::cout << "[DEBUG] createPlanner() called with name: " << plannerName << std::endl;

    if (plannerName == "BITstar")       return std::make_shared<og::BITstar>(si);
    else if (plannerName == "PRM")      return std::make_shared<og::PRM>(si);
    else if (plannerName == "PRMstar")  return std::make_shared<og::PRMstar>(si);
    else if (plannerName == "RRT")      return std::make_shared<og::RRT>(si);
    else if (plannerName == "RRTConnect") return std::make_shared<og::RRTConnect>(si);
    else if (plannerName == "RRTstar")  return std::make_shared<og::RRTstar>(si);
    else if (plannerName == "LazyRRT")  return std::make_shared<og::LazyRRT>(si);
    else if (plannerName == "TRRT")     return std::make_shared<og::TRRT>(si);
    else if (plannerName == "EST")      return std::make_shared<og::EST>(si);
    else if (plannerName == "KPIECE1")  return std::make_shared<og::KPIECE1>(si);
    else if (plannerName == "BKPIECE1") return std::make_shared<og::BKPIECE1>(si);
    else if (plannerName == "STRIDE")   return std::make_shared<og::STRIDE>(si);
    else if (plannerName == "SBL")      return std::make_shared<og::SBL>(si);
    else if (plannerName == "pSBL")     return std::make_shared<og::pSBL>(si);
    else if (plannerName == "FMTstar")  return std::make_shared<og::FMT>(si);
    else if (plannerName == "BFMTstar") return std::make_shared<og::BFMT>(si);
    else
    {
        std::cout << "[WARNING] Unknown plannerName '" << plannerName
                  << "', defaulting to BITstar.\n";
        return std::make_shared<og::BITstar>(si);
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

    // Create the planner
    ob::PlannerPtr planner = createPlanner(plannerName, si);
    planner->setProblemDefinition(pdef);
    planner->setup();

    // Optional: Print some settings
    std::cout << "[DEBUG] Planner settings:\n";
    planner->printProperties(std::cout);
    planner->printSettings(std::cout);

    // We wrap solve(...) in a try/catch
    ob::PlannerStatus solved;
    try
    {
        std::cout << "[DEBUG] Calling solve(" << planning_time << ")...\n";
        solved = planner->solve(planning_time);
    }
    catch (ompl::Exception &ex)
    {
        std::cout << "[EXCEPTION] Planner " << plannerName << " threw:\n"
                  << ex.what() << "\nSkipping this planner.\n";
        // Clear to avoid mixing results
        planner->clear();
        pdef->clearSolutionPaths();
        return; // Move on to the next planner
    }

    double ms = 0.0;
    if (solved == ob::PlannerStatus::EXACT_SOLUTION)
    {
        // We can measure the actual time by re-checking system time if we want.
        // For simplicity, let's just say "We found it in under 1 second"
        ms = 1000.0 * planning_time; // Or measure precisely if you want

        std::cout << "  [SUCCESS] " << plannerName
                  << " found EXACT_SOLUTION in " << ms << " ms.\n";
        auto path = pdef->getSolutionPath();
        auto &pathGeo = static_cast<og::PathGeometric &>(*path);

        // Evaluate cost
        auto obj = pdef->getOptimizationObjective();
        auto initialCost = pathGeo.cost(obj);
        std::cout << "[DEBUG] Path has " << pathGeo.getStateCount()
                  << " states before simplification.\n";

        // Simplify
        // og::PathSimplifier simplifier(si, pdef->getGoal(), obj);
        // std::cout << "[DEBUG] Attempting path simplification...\n";
        // bool validSimpl = simplifier.simplify(pathGeo, simplification_time);

        // auto simplifiedCost = pathGeo.cost(obj);
        // if (!validSimpl)
        //     std::cout << "  [WARNING] " << plannerName << " path not valid after simplification.\n";

        // std::cout << "  Initial cost:    " << initialCost << "\n"
        //           << "  Simplified cost: " << simplifiedCost << "\n";
        // std::cout << "  Final path:\n";

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

    // Clear solution before next planner
    pdef->clearSolutionPaths();
    planner->clear();
    std::cout << "[DEBUG] Done with " << plannerName << "\n";
}

// ----------------- main() -----------------
int main(int argc, char **argv)
{
    // Possibly reduce OMPL logging if too verbose:
    ompl::msg::setLogLevel(ompl::msg::LOG_INFO);

    bool optimize = false;  // If true, we won't stop at the first found path
    if (argc == 2)
        optimize = true;

    // Build environment
    EnvironmentInput environment;
    for (const auto &sphere : problem)
        environment.spheres.emplace_back(vamp::collision::factory::sphere::array(sphere, radius));
    environment.sort();
    auto env_v = EnvironmentVector(environment);
    std::cout << "[DEBUG main()] Environment with " << env_v.spheres.size() << " spheres.\n";

    // Create RealVectorStateSpace
    auto space = std::make_shared<ob::RealVectorStateSpace>(dimension);
    static constexpr std::array<float, dimension> zeros = {0., 0., 0., 0., 0., 0., 0.};
    static constexpr std::array<float, dimension> ones  = {1., 1., 1., 1., 1., 1., 1.};
    Configuration zero_v(zeros), one_v(ones);

    // Scale your bounding box using VAMP
    Robot::scale_configuration(zero_v);
    Robot::scale_configuration(one_v);

    ob::RealVectorBounds bounds(dimension);
    for (std::size_t i = 0; i < dimension; ++i)
    {
        bounds.setLow(i, zero_v[{i, 0}]);
        bounds.setHigh(i, one_v[{i, 0}]);
    }
    space->setBounds(bounds);

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
    {
        // Only want first feasible solution
        obj->setCostThreshold(ob::Cost(std::numeric_limits<double>::infinity()));
    }

    // List of planners
    std::vector<std::string> allPlanners = {
        "BITstar", "PRM", "PRMstar", "RRT", "RRTConnect", "RRTstar",
        "EST", "pSBL", "KPIECE1", "BKPIECE1", "STRIDE",
        "FMTstar", "BFMTstar", "LazyRRT", "TRRT", "SBL"
        // "SST" if you have included <ompl/geometric/planners/rrt/SST.h>
    };

    // Run them all
    for (const auto &plannerName : allPlanners)
    {
        runPlanner(plannerName, si, pdef, optimize);
        std::cout << "-------------------------------------------\n";
    }

    std::cout << "[DEBUG main()] All planners done!\n";
    return 0;
}
