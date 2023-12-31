/****************************************************************
 * Copyright (c) 2020~2022, 18-643 Course Staff, CMU
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:

 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.

 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.

 * The views and conclusions contained in the software and
 * documentation are those of the authors and should not be
 * interpreted as representing official policies, either expressed or
 * implied, of the FreeBSD Project.
 ****************************************************************/

/*
 * CMU 18643 Fall 2022 Lab Exercise
 *
 * You can edit this file
 */

/****************************************************************
 * Blocked convolution layer implementation
 * based on Figure 5:
 *    C. Zhang, et al., "Optimizing FPGA-based Accelerator
 *    Design for Deep Convolutional Neural Networks," FPGA, 2015.
 ****************************************************************/

#include "krnl_cnn_layer0.h"

// Prevent aliasing
#undef BATCH_SIZE

#include "util643.h"

void cnn0_blocked_kernel(cnndata_t BufI[TN_0][TR_0*S_WTS+K_WTS-S_WTS][TC_0*S_WTS+K_WTS-S_WTS],
                        cnndata_t BufO[TM_0][TR_0][TC_0],
                        cnndata_t BufW[TM_0][TN_0][K_WTS][K_WTS]);

#ifdef __VITIS_CL__
extern "C" {
#endif
void krnl_cnn_layer0(const cnndata_t* input, const cnndata_t* weights,
        cnndata_t* output, uint64_t batch_size) {

  index_t iter;
  index_t row, col, to, ti;

  cnndata_t BufI[TN_0][TR_0*S_WTS+K_WTS-S_WTS][TC_0*S_WTS+K_WTS-S_WTS];
  cnndata_t BufO[TM_0][TR_0][TC_0];
  cnndata_t BufW[TM_0][TN_0][K_WTS][K_WTS];

  for(iter = 0; iter < batch_size; iter++) {        // Batch Loop
    for(row = 0; row < R_OFM(0); row += TR_0) {     // Tiled Row Loop
      for(col = 0; col < C_OFM(0); col += TC_0) {   // Tiled Column Loop
        for(to = 0; to < M_OFM(0); to += TM_0) {    // Tiled Output Channel Loop
          // Temporary versions of incremented indices;
          // Same usage as in ZhangIsfpga_2()
          index_t trr, tcc, too, tii;

          // Only need to zero BufO in this loop ordering
          {
            // Indices internal to the block: count from 0
            index_t ioo, icc, irr;

            for(ioo = 0; ioo < TM_0; ioo++) {
              for(irr = 0; irr < TR_0; irr++) {
                for(icc = 0; icc < TC_0; icc++) {
                  BufO[ioo][irr][icc] = 0;
                }
              }
            }
          }

          // Tiled Input Channel Loop
          for(ti = 0; ti < N_IFM(0); ti += TN_0) {
            // Load active input feature map into local buffer
            {
              // Indices internal to the block: count from 0
              index_t irr, icc, iii;

              // Incremented temporary indices for input row and col
              index_t xrr, xcc;

              // Loop bounds
              index_t tii_max, xrr_max, xcc_max;
              tii_max = MIN(ti + TN_0, N_IFM(0));
              xrr_max = MIN(row + TR_0, R_OFM(0)) * S_WTS + K_WTS - S_WTS;
              xcc_max = MIN(col + TC_0, C_OFM(0)) * S_WTS + K_WTS - S_WTS;

              for(tii = ti, iii = 0; tii < tii_max; tii++, iii++) {
                for(xrr = row * S_WTS, irr = 0; xrr < xrr_max; xrr++, irr++) {
                  for(xcc = col * S_WTS, icc = 0; xcc < xcc_max; xcc++, icc++) {
                    BufI[iii][irr][icc] = ARRAYi_0(input, iter, tii, xrr, xcc,
                      batch_size, N_IFM(0), R_IFM(0), C_IFM(0));
                  }
                }
              }
            }

            // Load active weights into local buffer
            {
              // Indices internal to the block: count from 0
              index_t ioo, iii, irr, icc;

              // Loop bounds
              index_t too_max, tii_max;
              too_max = MIN(to + TM_0, M_OFM(0));
              tii_max = MIN(ti + TN_0, N_IFM(0));

              for(too = to, ioo = 0; too < too_max; too++, ioo++) {
                for(tii = ti, iii = 0; tii < tii_max; tii++, iii++) {
                  for(irr = 0; irr < K_WTS; irr++) {
                    for(icc = 0; icc < K_WTS; icc++) {
                      BufW[ioo][iii][irr][icc] = ARRAYw_0(weights, too, tii, irr,
                        icc, M_OFM(0), N_IFM(0), K_WTS, K_WTS);
                    }
                  }
                }

                /* Write 0s into over-run regions at the end;
                 * This way convolve_kernel() accumulates correctly
                 * without needing a special case
                 */
                if (iii < TN_0) {
                  for(; iii < TN_0; iii++) {
                    for(irr = 0; irr < K_WTS; irr++) {
                      for(icc = 0; icc < K_WTS; icc++) {
                        BufW[ioo][iii][irr][icc] = 0;
                      }
                    }
                  }
                }
              }
            }

            // Call the blocked cnn kernel
            cnn0_blocked_kernel(BufI, BufO, BufW);
          }

          // Unload finished active intermedaite output feature map from local
          // to full buffer
          {
            // Indices internal to the block: count from 0
            index_t ioo, icc, irr;

            // Loop bounds
            index_t too_max, tcc_max, trr_max;
            too_max = MIN(to + TM_0, M_OFM(0));
            tcc_max = MIN(col + TC_0, C_OFM(0));
            trr_max = MIN(row + TR_0, R_OFM(0));

            for(too = to, ioo = 0; too < too_max; too++, ioo++) {
              for(trr = row, irr = 0; trr < trr_max; trr++, irr++) {
                for(tcc = col, icc = 0; tcc < tcc_max; tcc++, icc++) {
                  ARRAYo_0(output, iter, too, trr, tcc, batch_size, M_OFM(0),
                    R_OFM(0), C_OFM(0)) = BufO[ioo][irr][icc];
                }
              }
            }
          }
        }
      }
    }
  }
}

#ifdef __VITIS_CL__ // for lab 3
} // extern
#endif

void cnn0_blocked_kernel(cnndata_t BufI[TN_0][TR_0*S_WTS+K_WTS-S_WTS][TC_0*S_WTS+K_WTS-S_WTS],
                        cnndata_t BufO[TM_0][TR_0][TC_0],
                        cnndata_t BufW[TM_0][TN_0][K_WTS][K_WTS]) {
  index_t to_b, ti_b, row_b, col_b;
  index_t i, j;

  Krow: for(i = 0; i < K_WTS; i++) {
  Kcol: for(j = 0; j < K_WTS; j++) {
  Row: for(row_b = 0; row_b < TR_0; row_b++) {
    Col: for(col_b = 0; col_b < TC_0; col_b++) {
      To: for(to_b = 0; to_b < TM_0; to_b++) {
        Ti: for(ti_b = 0; ti_b < TN_0; ti_b++) {
              BufO[to_b][row_b][col_b]+= BufW[to_b][ti_b][i][j]*
                BufI[ti_b][S_WTS*row_b+i][S_WTS*col_b+j];
            }
          }
        }
      }
    }
  }
}
