// Copyright (c) 2016-2017, Philipp Sebastian Ruppel
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universität Hamburg nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <stddef.h>  // for size_t

#include <bio_ik/forward_kinematics.hpp>  // for RobotFK, RobotFK_MoveIt
#include <bio_ik/ik_base.hpp>             // for IKBase, IKFactory
#include <bio_ik/problem.hpp>             // for Problem
#include <bio_ik/utils.hpp>               // for aligned_vector, LOG, IKParams
#include <memory>                         // for allocator_traits<>::value_type
#include <vector>                         // for vector

#include "bio_ik/frame.hpp"       // for Frame, Quaternion, Vector3
#include "bio_ik/robot_info.hpp"  // for RobotInfo

namespace bio_ik {

struct IKTest : IKBase {
  RobotFK_MoveIt fkref_;

  std::vector<double> temp_;

  double d_rot_sum_, d_pos_sum_, d_div_;

  IKTest(const IKParams& params) : IKBase(params), fkref_(params.robot_model) {
    d_rot_sum_ = d_pos_sum_ = d_div_ = 0;
  }

  /*double tipdiff(const std::vector<Frame>& fa, const std::vector<Frame>& fb)
  {
      double diff = 0.0;
      for(size_t i = 0; i < problem_.tip_link_indices.size(); i++)
      {
          //LOG_VAR(fa[i]);
          //LOG_VAR(fb[i]);
          diff += fa[i].rot.angleShortestPath(fb[i].rot);
          diff += fa[i].pos.distance(fb[i].pos);
      }
      return diff;
  }*/

  void initialize(const Problem& problem) {
    IKBase::initialize(problem);

    fkref_.initialize(problem_.tip_link_indices);
    model_.initialize(problem_.tip_link_indices);

    fkref_.applyConfiguration(problem_.initial_guess);
    model_.applyConfiguration(problem_.initial_guess);

    // double diff = tipdiff(fkref_.getTipFrames(), model_.getTipFrames());
    // LOG_VAR(diff);

    /*{
        auto& fa = fkref_.getTipFrames();
        auto& fb = model_.getTipFrames();
        for(size_t i = 0; i < problem_.tip_link_indices.size(); i++)
        {
            LOG("d rot", i, fa[i].rot.angleShortestPath(fb[i].rot));
            LOG("d pos", i, fa[i].pos.distance(fb[i].pos));
        }
    }*/

    {
      temp_ = problem_.initial_guess;
      for (size_t ivar : problem_.active_variables)
        if (modelInfo_.isRevolute(ivar) || modelInfo_.isPrismatic(ivar))
          temp_[ivar] = modelInfo_.clip(temp_[ivar] + random(-0.1, 0.1), ivar);

      fkref_.applyConfiguration(temp_);
      auto& fa = fkref_.getTipFrames();

      model_.applyConfiguration(problem_.initial_guess);
      model_.initializeMutationApproximator(problem_.active_variables);

      std::vector<aligned_vector<Frame>> fbm;

      std::vector<double> mutation_values;
      for (size_t ivar : problem_.active_variables)
        mutation_values.push_back(temp_[ivar]);
      const double* mutation_ptr = mutation_values.data();

      model_.computeApproximateMutations(1, &mutation_ptr, fbm);

      auto& fb = fbm[0];

      // auto& fb = model_.getTipFrames();

      for (size_t i = 0; i < problem_.tip_link_indices.size(); i++) {
        // LOG("d rot", i, fa[i].rot.angleShortestPath(fb[i].rot));
        // LOG("d pos", i, fa[i].pos.distance(fb[i].pos));

        d_rot_sum_ += fa[i].rot.angleShortestPath(fb[i].rot);
        d_pos_sum_ += fa[i].pos.distance(fb[i].pos);
        d_div_ += 1;
      }
    }

    LOG("d rot avg", d_rot_sum_ / d_div_);
    LOG("d pos avg", d_pos_sum_ / d_div_);
  }

  void step() {}

  const std::vector<double>& getSolution() const {
    return problem_.initial_guess;
  }
};

static IKFactory::Class<IKTest> test("test");
}  // namespace bio_ik
