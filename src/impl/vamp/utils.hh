#pragma once

#include <chrono>
#include <cstdlib>
#include <utility>
#include <cstdint>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace vamp::utils
{
    inline constexpr auto round_size(std::size_t size, std::size_t block) noexcept -> std::size_t
    {
        return ((size + block - 1) / block) * block;
    }

    inline bool is_aligned(const void *ptr, uintptr_t alignment) noexcept
    {
        auto iptr = reinterpret_cast<uintptr_t>(ptr);
        return not (iptr % alignment);
    }

    // MSVC UCRT has no aligned_alloc; use _aligned_malloc and pair with vector_free
    // (_aligned_free). POSIX aligned_alloc pairs with free — vector_free abstracts that
    // so unique_ptr deleters stay correct on Windows clang-cl.
    template <typename T, std::size_t alignment, std::size_t vector_size>
    inline auto vector_alloc(std::size_t n) noexcept -> T *
    {
        const std::size_t bytes = sizeof(T) * round_size(n, vector_size);
#if defined(_WIN32)
        // _aligned_malloc(size, alignment) — argument order is opposite aligned_alloc.
        return static_cast<T *>(_aligned_malloc(bytes, alignment));
#else
        return static_cast<T *>(aligned_alloc(alignment, bytes));
#endif
    }

    inline void vector_free(void *ptr) noexcept
    {
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

    // Because ceil isn't constexpr until C++23 for some reason.....
    inline constexpr auto c_ceil(double d) noexcept -> std::size_t
    {
        const auto s = static_cast<std::size_t>(d);
        return d > s ? s + 1 : s;
    }

    // Same deal with div()...
    inline constexpr auto c_div(std::size_t idx, std::size_t dim) noexcept
        -> std::pair<std::size_t, std::size_t>
    {
        return {idx / dim, idx % dim};
    }

    inline auto get_elapsed_nanoseconds(const std::chrono::time_point<std::chrono::steady_clock> &start)
        -> std::size_t
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
            .count();
    }
}  // namespace vamp::utils
