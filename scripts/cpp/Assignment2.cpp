#include <chrono>
#include <vamp/vector.hh>
#include <vamp/planning/utils.hh>
#include <vamp/collision/factory.hh>
#include <vamp/robots/panda.hh>
#include <vamp/planning/validate.hh>

static constexpr auto dimension = 7;
static constexpr const std::size_t rake = vamp::FloatVectorWidth; // Normally it's 8
static constexpr const std::size_t resolution = 32; // Resolution for back-stepping.
using Robot = vamp::robots::Panda;
using Configuration = Robot::Configuration;
using EnvironmentInput = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;

// Spheres for the cage problem - (x, y, z) center coordinates with fixed, common radius defined below
static const std::vector<std::array<float, 3>> problem = {
    {0.55, 0, 0.25},
    {0.35, 0.35, 0.25},
    {0, 0.55, 0.25},
    {-0.55, 0, 0.25},
    {-0.35, -0.35, 0.25},
    // {0, -0.55, 0.25},
    // {0.35, -0.35, 0.25},
    // {0.35, 0.35, 0.8},
    // {0, 0.55, 0.8},
    // {-0.35, 0.35, 0.8},
    // {-0.55, 0, 0.8},
    // {-0.35, -0.35, 0.8},
    // {0, -0.55, 0.8},
    // {0.35, -0.35, 0.8},
};
// Radius for obstacle spheres
static constexpr float radius = 0.2;



inline auto collision_checking(
        const Configuration &start,
        const Configuration &goal,
        const EnvironmentVector &environment) -> bool
    {
        // Generate percentage increments for 8 samples
    
        const auto percents = vamp::FloatVector<rake>(vamp::planning::Percents<rake>::percents);

        // Linear interpolation between q0 and q1
        vamp::robots::Panda::ConfigurationBlock<rake> block;

        auto vector = goal - start;
        for (auto i = 0U; i < Robot::dimension; ++i) {
            block[i] = start.broadcast(i) + (vector.broadcast(i) * percents);
        }

        // Print generated configurations
        // std::cout << "Generated configurations:\n";
        // for (std::size_t i = 0; i < rake; ++i) {
        //     std::cout << "Sample " << i + 1 << ": " << block[i] << "\n";
        // }

        // Collision check all samples using fkcc
        // bool is_collision_free = Robot::template fkcc<rake>(environment, block);
        // if (!is_collision_free) {
        //     throw std::runtime_error("Collision detected in one or more samples.");
        //     return false;
        // }
        float distance = 1/32;
        const std::size_t n = std::max(std::ceil((distance) / static_cast<float>(rake) * resolution), 1.F);


        bool valid = (environment.attachments) ? Robot::template fkcc_attach<rake>(environment, block) :
                                                    Robot::template fkcc<rake>(environment, block);
        if (not valid)
        {
            return valid;
        }

        const auto backstep = vector / (rake * n);
        for (auto i = 1U; i < n; ++i)
        {
            for (auto j = 0U; j < Robot::dimension; ++j)
            {
                block[j] = block[j] - backstep.broadcast(j);
            }

            if (not Robot::template fkcc<rake>(environment, block))
            {
                return false;
            }
        }
    
        return true;

    }


auto main(int argc, char **) -> int
{
    // Build sphere cage environment
    EnvironmentInput environment;
    for (const auto &sphere : problem)
    {
        environment.spheres.emplace_back(vamp::collision::factory::sphere::array(sphere, radius));
    }

    environment.sort();
    auto env_v = EnvironmentVector(environment);

    // q0 and q1 configurations
    std::array<float, dimension>  q0_array = {0., -0.785, 0., -2.356, 0., 1.571, 0.785};
    std::array<float, dimension>  q1_array = {2.35, 1., 0., -0.8, 0, 2.5, 0.785};
    Configuration q0(q0_array), q1(q1_array);
    // This will print out an array of two SIMD vector with zero padding at the end
    std::cout << q0 << std::endl << q1 << std::endl;

    auto start_time_vamp = std::chrono::steady_clock::now();
    auto cc = collision_checking(q0, q1, env_v);
    auto elapsed_time_vamp = vamp::utils::get_elapsed_nanoseconds(start_time_vamp);
    std::cout << "Collision free? " << cc << std::endl;
    std::cout << "Collision checking time with VAMP: " << elapsed_time_vamp << "nanoseconds"<< std::endl;
    return 0;
}