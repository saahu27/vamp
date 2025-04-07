#include <chrono>
#include <vamp/vector.hh>
#include <vamp/planning/utils.hh>

static constexpr auto dimension = 12;
using Configuration = vamp::FloatVector<dimension>;
auto main(int argc, char **) -> int
{

    // q0 and q1 configurations
    std::array<float, dimension>  q0_array = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::array<float, dimension>  q1_array = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5};
    Configuration q0(q0_array), q1(q1_array);
    // This will print out an array of two SIMD vector with zero padding at the end
    std::cout << q0 << std::endl << q1 << std::endl;

    std::array<float, dimension> qresultadd{0};
    std::array<float, dimension> qresultsub{0};
    std::array<float, dimension> qresultdiv{0};
    std::array<float, dimension> qresultmul{0};
    auto start_time = std::chrono::steady_clock::now();
    for (size_t i = 0; i < q0_array.size(); i++)
    {
        qresultadd[i] = q0_array[i] + q1_array[i];
        qresultsub[i] = q0_array[i] - q1_array[i];
        qresultdiv[i] = q0_array[i] / q1_array[i];
        qresultmul[i] = q0_array[i] * q1_array[i];
        //
        // Add your code here with q0_array and q1_array
        //
    }
    auto elapsed_time = vamp::utils::get_elapsed_nanoseconds(start_time);
    std::cout << "Elapsed time with vector: " << elapsed_time << "nanoseconds"<< std::endl;
    //std::cout << "Result: " << qresult << std::endl;

    auto start_time_vamp = std::chrono::steady_clock::now();
    Configuration qresultadd_vamp;
    Configuration qresultsub_vamp;
    Configuration qresultdiv_vamp;
    Configuration qresultmul_vamp;

    //
    // Add your code here with q0 and q1
    //
    qresultadd_vamp = q0 + q1;
    qresultsub_vamp = q0 - q1;
    qresultdiv_vamp = q0 / q1;
    qresultmul_vamp = q0 * q1;


    auto elapsed_time_vamp = vamp::utils::get_elapsed_nanoseconds(start_time_vamp);
    std::cout << "VAMP ADD" << qresultadd_vamp << std::endl;
    std::cout << "VAMP SUB" << qresultsub_vamp << std::endl;
    std::cout << "VAMP DIV" << qresultdiv_vamp << std::endl;
    std::cout << "VAMP MUL" << qresultmul_vamp << std::endl;
    std::cout << "Elapsed time with VAMP: " << elapsed_time_vamp << "nanoseconds"<< std::endl;
    return 0;
}