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
#include <cmath>               // for isfinite
#include <cstddef>             // for size_t
#include <ext/alloc_traits.h>  // for __alloc_traits<>::value_type
#include <kdl/frames.hpp>      // for Twist, Vector
#include <vector>              // for vector, allocator

#include "bio_ik/frame.hpp"       // for Frame, frameTwist
#include "bio_ik/robot_info.hpp"  // for RobotInfo
#include "ik_base.hpp"            // for IKFactory, IKBase
#include "problem.hpp"            // for Problem, Problem::GoalInfo
#include "utils.hpp"              // for FNPROFILER

namespace bio_ik {

// pseudoinverse jacobian solver
// (mainly for testing RobotFK_Jacobian::computeJacobian)
template <class BASE>
struct IKJacobianBase : BASE {
  // IKBase functions
  using BASE::computeFitness;

  // IKBase variables
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

  void optimizeJacobian(std::vector<double>& solution) {
    FNPROFILER();

    Eigen::Index tip_count =
        static_cast<Eigen::Index>(problem_.tip_link_indices.size());
    tip_diffs_.resize(tip_count * 6);
    joint_diffs_.resize(
        static_cast<Eigen::Index>(problem_.active_variables.size()));

    // compute fk
    model_.applyConfiguration(solution);

    double translational_scale = 1;
    double rotational_scale = 1;

    // compute goal diffs
    tip_frames_temp_ = model_.getTipFrames();
    for (Eigen::Index itip = 0; itip < tip_count; ++itip) {
      auto twist = frameTwist(tip_frames_temp_[static_cast<size_t>(itip)],
                              tipObjectives_[static_cast<size_t>(itip)]);
      tip_diffs_(itip * 6 + 0) = twist.vel.x() * translational_scale;
      tip_diffs_(itip * 6 + 1) = twist.vel.y() * translational_scale;
      tip_diffs_(itip * 6 + 2) = twist.vel.z() * translational_scale;
      tip_diffs_(itip * 6 + 3) = twist.rot.x() * rotational_scale;
      tip_diffs_(itip * 6 + 4) = twist.rot.y() * rotational_scale;
      tip_diffs_(itip * 6 + 5) = twist.rot.z() * rotational_scale;
    }

    // compute jacobian
    {
      model_.computeJacobian(problem_.active_variables, jacobian_);
      Eigen::Index icol = 0;
      for (auto __attribute__((unused)) _ : problem_.active_variables) {
        for (Eigen::Index itip = 0; itip < tip_count; ++itip) {
          jacobian_(itip * 6 + 0, icol) *= translational_scale;
          jacobian_(itip * 6 + 1, icol) *= translational_scale;
          jacobian_(itip * 6 + 2, icol) *= translational_scale;
          jacobian_(itip * 6 + 3, icol) *= rotational_scale;
          jacobian_(itip * 6 + 4, icol) *= rotational_scale;
          jacobian_(itip * 6 + 5, icol) *= rotational_scale;
        }
        icol++;
      }
    }

    // get pseudoinverse and multiply
    joint_diffs_ =
        jacobian_.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
            .solve(tip_diffs_);
    // joint_diffs_ = (jacobian_.transpose() *
    // jacobian_).ldlt().solve(jacobian_.transpose() * tip_diffs_);

    // apply joint deltas and clip
    {
      int icol = 0;
      for (auto ivar : problem_.active_variables) {
        auto v = solution[ivar] + joint_diffs_(icol);
        if (!std::isfinite(v)) continue;
        v = modelInfo_.clip(v, ivar);
        solution[ivar] = v;
        icol++;
      }
    }
  }
};

// simple gradient descent
template <int if_stuck, size_t threads>
struct IKGradientDescent : IKBase {
  std::vector<double> solution_, best_solution_, gradient_, temp_;
  bool reset_;

  IKGradientDescent(const IKParams& p) : IKBase(p) {}

  void initialize(const Problem& problem) {
    IKBase::initialize(problem);
    solution_ = problem_.initial_guess;
    if (thread_index_ > 0)
      for (auto& vi : problem_.active_variables)
        solution_[vi] = random(modelInfo_.getMin(vi), modelInfo_.getMax(vi));
    best_solution_ = solution_;
    reset_ = false;
  }

  const std::vector<double>& getSolution() const { return best_solution_; }

  void step() {
    // random reset if stuck
    if (reset_) {
      reset_ = false;
      for (auto& vi : problem_.active_variables)
        solution_[vi] = random(modelInfo_.getMin(vi), modelInfo_.getMax(vi));
    }

    // compute gradient_ direction
    temp_ = solution_;
    double jd = 0.0001;
    gradient_.resize(solution_.size(), 0);
    for (auto ivar : problem_.active_variables) {
      temp_[ivar] = solution_[ivar] - jd;
      double p1 = computeFitness(temp_);

      temp_[ivar] = solution_[ivar] + jd;
      double p3 = computeFitness(temp_);

      temp_[ivar] = solution_[ivar];

      gradient_[ivar] = p3 - p1;
    }

    // normalize gradient direction
    double sum = 0.0001;
    for (auto ivar : problem_.active_variables) sum += fabs(gradient_[ivar]);
    double f = 1.0 / sum * jd;
    for (auto ivar : problem_.active_variables) gradient_[ivar] *= f;

    // initialize line search
    temp_ = solution_;

    for (auto ivar : problem_.active_variables)
      temp_[ivar] = solution_[ivar] - gradient_[ivar];
    double p1 = computeFitness(temp_);

    // for(auto ivar : problem_.active_variables) temp_[ivar] = solution_[ivar];
    // double p2 = computeFitness(temp_);

    for (auto ivar : problem_.active_variables)
      temp_[ivar] = solution_[ivar] + gradient_[ivar];
    double p3 = computeFitness(temp_);

    double p2 = (p1 + p3) * 0.5;

    // linear step size estimation
    double cost_diff = (p3 - p1) * 0.5;
    double joint_diff = p2 / cost_diff;

    // in case cost_diff is 0
    if (!std::isfinite(joint_diff)) joint_diff = 0.0;

    // apply optimization step
    // (move along gradient direction by estimated step size)
    for (auto ivar : problem_.active_variables)
      temp_[ivar] =
          modelInfo_.clip(solution_[ivar] - gradient_[ivar] * joint_diff, ivar);

    if (if_stuck == 'c') {
      // always accept solution and continue
      solution_ = temp_;
    } else {
      // has solution improved?
      if (computeFitness(temp_) < computeFitness(solution_)) {
        // solution improved -> accept solution
        solution_ = temp_;
      } else {
        if (if_stuck == 'r') {
          // reset if stuck
          reset_ = true;
        }
      }
    }

    // update best solution
    if (computeFitness(solution_) < computeFitness(best_solution_))
      best_solution_ = solution_;
  }

  size_t concurrency() const { return threads; }
};

static IKFactory::Class<IKGradientDescent<' ', 1>> gd("gd");
static IKFactory::Class<IKGradientDescent<' ', 2>> gd_2("gd_2");
static IKFactory::Class<IKGradientDescent<' ', 4>> gd_4("gd_4");
static IKFactory::Class<IKGradientDescent<' ', 8>> gd_8("gd_8");

static IKFactory::Class<IKGradientDescent<'r', 1>> gd_r("gd_r");
static IKFactory::Class<IKGradientDescent<'r', 2>> gd_2_r("gd_r_2");
static IKFactory::Class<IKGradientDescent<'r', 4>> gd_4_r("gd_r_4");
static IKFactory::Class<IKGradientDescent<'r', 8>> gd_8_r("gd_r_8");

static IKFactory::Class<IKGradientDescent<'c', 1>> gd_c("gd_c");
static IKFactory::Class<IKGradientDescent<'c', 2>> gd_2_c("gd_c_2");
static IKFactory::Class<IKGradientDescent<'c', 4>> gd_4_c("gd_c_4");
static IKFactory::Class<IKGradientDescent<'c', 8>> gd_8_c("gd_c_8");

// pseudoinverse jacobian only
template <size_t threads>
struct IKJacobian : IKJacobianBase<IKBase> {
  using IKBase::initialize;
  std::vector<double> solution_;
  IKJacobian(const IKParams& params) : IKJacobianBase<IKBase>(params) {}
  void initialize(const Problem& problem) {
    IKJacobianBase<IKBase>::initialize(problem);
    solution_ = problem_.initial_guess;
    if (thread_index_ > 0)
      for (auto& vi : problem_.active_variables)
        solution_[vi] = random(modelInfo_.getMin(vi), modelInfo_.getMax(vi));
  }
  const std::vector<double>& getSolution() const { return solution_; }
  void step() { optimizeJacobian(solution_); }
  size_t concurrency() const { return threads; }
};
static IKFactory::Class<IKJacobian<1>> jac("jac");
static IKFactory::Class<IKJacobian<2>> jac_2("jac_2");
static IKFactory::Class<IKJacobian<4>> jac_4("jac_4");
static IKFactory::Class<IKJacobian<8>> jac_8("jac_8");
}  // namespace bio_ik
