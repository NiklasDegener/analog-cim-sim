/******************************************************************************
 * Copyright (C) 2025 Rebecca Pelke                                           *
 * All Rights Reserved                                                        *
 *                                                                            *
 * This is work is licensed under the terms described in the LICENSE file     *
 * found in the root directory of this source tree.                           *
 ******************************************************************************/
#include "mapping/int_mapper/int_ii.h"
#include "helper/config.h"

namespace nq {

MapperIntII::MapperIntII() :
    vd_p_(CFG.N, 0), tmp_out_int_(CFG.M * CFG.SPLIT.size(), 0),
    tmp_out_fp_(CFG.M * CFG.SPLIT.size(), 0.0), Mapper(true) {}

MapperIntII::~MapperIntII() {}

void MapperIntII::d_write(const int32_t *mat, int32_t m_matrix,
                          int32_t n_matrix) {
    d_write_diff(mat, m_matrix, n_matrix);
}

void MapperIntII::a_write(int32_t m_matrix, int32_t n_matrix) {
    a_write_p_m(m_matrix, n_matrix);
}

void MapperIntII::d_mvm(int32_t *res, const int32_t *vec, const int32_t *mat,
                        int32_t m_matrix, int32_t n_matrix) {
    // The splitted matrix is of size CFG.SPLITsize*M x N (CFG.SPLITsize values
    // per original matrix value) Two matrices exist: gd+ (gd_p_) and gd-
    // (gd_m_) The input (which is signed) is shifted to the positive domain
    const std::vector<uint32_t> &split = CFG.SPLIT;
    const uint32_t tmp_size = m_matrix * split.size();
    std::fill(tmp_out_int_.begin(), tmp_out_int_.end(), 0);

    // Shift input bits to positive range (+ 2^(B-1))
    for (size_t n = 0; n < n_matrix; ++n) {
        vd_p_[n] = (1 << (CFG.I_BIT - 1)) + vec[n];
    }

    for (size_t t_m = 0; t_m < tmp_size; ++t_m) {
        for (size_t n = 0; n < n_matrix; ++n) {
            tmp_out_int_[t_m] += (gd_p_[t_m][n] - gd_m_[t_m][n]) * vd_p_[n];
        }
    }

    for (size_t m = 0; m < m_matrix; ++m) {
        for (size_t s = 0; s < split.size(); ++s) {
            res[m] += tmp_out_int_[m * split.size() + s] << shift_[s];
        }
        // Subtract term of compile-time constant
        res[m] -= ((sum_w_)[m] << (CFG.I_BIT - 1));
    }
}

void MapperIntII::d_mmm(int32_t *res, const int32_t *mat_A, const int32_t *mat_B,
                        int32_t m_matrix, int32_t k_matrix, int32_t n_matrix) {
    // The splitted matrix is of size CFG.SPLITsize*M x N (CFG.SPLITsize values
    // per original matrix value) Two matrices exist: gd+ (gd_p_) and gd-
    // (gd_m_) The input (which is signed) is shifted to the positive domain
    const std::vector<uint32_t> &split = CFG.SPLIT;
    const uint32_t tmp_size = m_matrix * split.size();

    // Split matrix columnwise and do MVM
    for (size_t n = 0; n < n_matrix; n++) {
        std::fill(tmp_out_int_.begin(), tmp_out_int_.end(), 0);

        // Shift input bits to positive range (+ 2^(B-1))
        for (size_t k = 0; k < k_matrix; ++k) {
            vd_p_[k] = (1 << (CFG.I_BIT - 1)) + mat_B[k * n_matrix + n];
        }
        for (size_t t_m = 0; t_m < tmp_size; ++t_m) {
            for (size_t k = 0; k < k_matrix; ++k) {
                tmp_out_int_[t_m] += (gd_p_[t_m][k] - gd_m_[t_m][k]) * vd_p_[k];
            }
        }

        for (size_t m = 0; m < m_matrix; ++m) {
            for (size_t s = 0; s < split.size(); ++s) {
                int temp = tmp_out_int_[m * split.size() + s] << shift_[s];
                res[m * n_matrix + n] += temp;
            }
            // Subtract term of compile-time constant
            res[m * n_matrix + n] -= ((sum_w_)[m] << (CFG.I_BIT - 1));
        }
    }
}

extern "C" void a_mmm_launch(int32_t *res, const float *mat_A_p,
                      const float *mat_A_m, const int32_t *mat_B, int m, int k,
                      int n);

void MapperIntII::a_mmm(int32_t *res, const int32_t *mat_A, const int32_t *mat_B,
                        int32_t m_matrix, int32_t k_matrix, int32_t n_matrix) {

    // Convert ia_p/m vectors into float arrays (probably expensive)
    int rows = ia_p_.size();
    int cols = ia_p_[0].size();

    std::vector<float> contiguous_p;
    std::vector<float> contiguous_m;
    contiguous_p.reserve(rows * cols); // rows * cols = 3072 = 32 * 32 * 3(Split)
    contiguous_m.reserve(rows * cols);

    // Only copy relevant parts of xbar (rest 0 anyway and would break indexing in kernel)
    for (int i = 0; i < std::min(rows, 3 * m_matrix); i++) {
        for (int j = 0; j < std::min(cols, k_matrix); j++) {
            std::cout << "i: " << i << ", j: " << j << ": A_p: " << ia_p_[i][j] << std::endl;
            contiguous_p.push_back(ia_p_[i][j]);
            contiguous_m.push_back(ia_m_[i][j]);
        }
    }

    float* ptr_p = contiguous_p.data();
    float* ptr_m = contiguous_m.data();

    // Dispatch kernel execution
    float resf[m_matrix * n_matrix] = {0.0f};
    std::cout << "Sum_w: " << std::endl;
    for (auto w : sum_w_) {
        std::cout << ", " << w << std::endl;
    }

    std::cout << "Res before launch:" << std::endl;
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            std::cout << ", " << res[i * 3 +j];
        }
    }

    a_mmm_launch(res, ptr_p, ptr_m, mat_B, m_matrix, k_matrix, n_matrix); // Split not included 
}

/*
void MapperIntII::a_mmm(int32_t *res, const int32_t *mat_A, const int32_t *mat_B,
                        int32_t m_matrix, int32_t k_matrix, int32_t n_matrix) {
    // The splitted matrix is of size CFG.SPLITsize*M x N (CFG.SPLITsize values
    // per original matrix value) Two matrices exist: ia+ (ia_p_) and ia-
    // (ia_m_).
    const std::vector<uint32_t> &split = CFG.SPLIT;
    const uint32_t tmp_size = m_matrix * CFG.SPLIT.size();
    for (size_t n = 0; n < n_matrix; n++) {
        std::fill(tmp_out_fp_.begin(), tmp_out_fp_.end(), 0.0);

        // Shift input bits to positive range (+ 2^(B-1))
        for (size_t k = 0; k < k_matrix; ++k) {
            vd_p_[k] = (1 << (CFG.I_BIT - 1)) + mat_B[k * n_matrix + n];
        }

        // For each bit in vd_p execute one MVM operation with ia_p_ and one with
        // ia_m_ MSB of input has position: CFG.I_BIT + 1 Subract both results in
        // the analog domain
        for (size_t i_bit = 0; i_bit < CFG.I_BIT + 1; ++i_bit) { //TO-DO: Nur bis I_BIT laufen lassen + checks einführen, dass B beim hochshift nicht overflowt
            // Calculcate multiplications with negative and positive weights
            for (size_t t_m = 0; t_m < tmp_size; ++t_m) {
                for (size_t k = 0; k < k_matrix; ++k) {
                    tmp_out_fp_[t_m] +=
                        (ia_p_[t_m][k] - ia_m_[t_m][k]) * ((vd_p_[k] >> i_bit) & 1); // Is this distributable, so can it be reduced to a single shift? Maybe not benefitial
                }
            }

            // Addition of the partial results caused by splitted weights
            for (size_t m = 0; m < m_matrix; ++m) {
                for (size_t s = 0; s < split.size(); ++s) {
                    res[m * n_matrix + n] += static_cast<int32_t>(
                        round(adc_->analog_digital_conversion(
                                tmp_out_fp_[m * split.size() + s]) /
                            i_step_size_[s] * std::pow(2, shift_[s]) *
                            std::pow(2, i_bit)));
                }
            }

            // Reset tmp_out vector
            std::fill(tmp_out_fp_.begin(), tmp_out_fp_.end(), 0);
        }

        // Subtract term of compile-time constant
        for (size_t m = 0; m < m_matrix; ++m) {
            res[m * n_matrix + n] -= ((sum_w_)[m] << (CFG.I_BIT - 1));
        }
    }
}*/

void MapperIntII::a_mvm(int32_t *res, const int32_t *vec, const int32_t *mat,
                        int32_t m_matrix, int32_t n_matrix) {
    // The splitted matrix is of size CFG.SPLITsize*M x N (CFG.SPLITsize values
    // per original matrix value) Two matrices exist: ia+ (ia_p_) and ia-
    // (ia_m_).
    const std::vector<uint32_t> &split = CFG.SPLIT;
    const uint32_t tmp_size = m_matrix * CFG.SPLIT.size();
    std::fill(tmp_out_fp_.begin(), tmp_out_fp_.end(), 0.0);

    // Shift input bits to positive range (+ 2^(B-1))
    for (size_t n = 0; n < n_matrix; ++n) {
        vd_p_[n] = (1 << (CFG.I_BIT - 1)) + vec[n];
    }

    // For each bit in vd_p execute one MVM operation with ia_p_ and one with
    // ia_m_ MSB of input has position: CFG.I_BIT + 1 Subract both results in
    // the analog domain
    for (size_t i_bit = 0; i_bit < CFG.I_BIT + 1; ++i_bit) {
        // Calculcate multiplications with negative and positive weights
        for (size_t t_m = 0; t_m < tmp_size; ++t_m) {
            for (size_t n = 0; n < n_matrix; ++n) {
                tmp_out_fp_[t_m] +=
                    (ia_p_[t_m][n] - ia_m_[t_m][n]) * ((vd_p_[n] >> i_bit) & 1);
            }
        }

        // Addition of the partial results caused by splitted weights
        for (size_t m = 0; m < m_matrix; ++m) {
            for (size_t s = 0; s < split.size(); ++s) {
                res[m] += static_cast<int32_t>(
                    round(adc_->analog_digital_conversion(
                              tmp_out_fp_[m * split.size() + s]) /
                          i_step_size_[s] * std::pow(2, shift_[s]) *
                          std::pow(2, i_bit)));
            }
        }

        // Reset tmp_out vector
        std::fill(tmp_out_fp_.begin(), tmp_out_fp_.end(), 0);
    }

    // Subtract term of compile-time constant
    for (size_t m = 0; m < m_matrix; ++m) {
        res[m] -= ((sum_w_)[m] << (CFG.I_BIT - 1));
    }
}

} // namespace nq