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

#include <Eigen/Core>          // For NumTraits
#include <bio_ik/ik_base.hpp>  // for IKSolver
#include <bio_ik/problem.hpp>  // for Problem, Problem::GoalInfo
#include <bio_ik/utils.hpp>    // for FNPROFILER
#include <cmath>               // for isfinite
#include <cstddef>             // for size_t
#include <ext/alloc_traits.h>  // for __alloc_traits<>::value_type
#include <kdl/frames.hpp>      // for Twist, Vector
#include <memory>
#include <optional>
#include <vector>  // for vector, allocator

#include "bio_ik/frame.hpp"       // for Frame, frameTwist
#include "bio_ik/robot_info.hpp"  // for RobotInfo

namespace bio_ik {

// simple gradient descent
template <int IF_STRUCK, size_t N_THREADS>
struct IKGradientDescent : IKSolver {
  std::vector<double> solution_, best_solution_, gradient_, temp_;
  bool reset_;

  IKGradientDescent(const IKParams& p) : IKSolver(p) {}

  void initialize(const Problem& problem);

  const std::vector<double>& getSolution() const { return best_solution_; }

  void step();

  size_t concurrency() const { return N_THREADS; }
};

// pseudoinverse jacobian solver
// (mainly for testing RobotFK_Jacobian::computeJacobian)
template <class BASE>
struct IKJacobianBase : BASE {
  // IKSolver functions
  using BASE::computeFitness;

  // IKSolver variables
  using BASE::model_;
  using BASE::modelInfo_;
  using BASE::params_;
  using BASE::problem_;

  std::vector<Frame> tipObjectives_;

  Eigen::VectorXd tip_diffs_;
  Eigen::VectorXd joint_diffs_;
  Eigen::MatrixXd jacobian_;
  std::vector<Frame> tip_frames_temp_;

  IKJacobianBase(const IKParams& p) : BASE(p) {}
  void initialize(const Problem& problem) {
    BASE::initialize(problem);
    tipObjectives_.resize(problem.tip_link_indices.size());
    for (auto& goal : problem.goals)
      tipObjectives_[goal.tip_index] = goal.frame;
  }
  void optimizeJacobian(std::vector<double>& solution);
};

// pseudoinverse jacobian only
template <size_t N_THREADS>
struct IKJacobian : IKJacobianBase<IKSolver> {
  using IKSolver::initialize;
  std::vector<double> solution_;
  IKJacobian(const IKParams& params) : IKJacobianBase<IKSolver>(params) {}
  void initialize(const Problem& problem);
  const std::vector<double>& getSolution() const { return solution_; }
  void step() { optimizeJacobian(solution_); }
  size_t concurrency() const { return N_THREADS; }
};

std::optional<std::unique_ptr<IKSolver>> makeGradientDecentSolver(
    const IKParams& params);

}  // namespace bio_ik
