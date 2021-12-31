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

#include <math.h>                            // for sqrt, M_PI
#include <moveit/robot_model/joint_model.h>  // for JointModel
#include <moveit/robot_model/link_model.h>   // for LinkModel
#include <moveit/robot_model/robot_model.h>  // for RobotModelConstPtr, Robo...
#include <stddef.h>                          // for size_t

#include <algorithm>                      // for remove, sort
#include <bio_ik/forward_kinematics.hpp>  // for RobotFK
#include <bio_ik/ik_base.hpp>             // for IKBase, IKFactory
#include <bio_ik/problem.hpp>             // for Problem
#include <bio_ik/utils.hpp>               // for mix, FNPROFILER, IKParams
#include <ext/alloc_traits.h>             // for __alloc_traits<>::value_...
#include <memory>                         // for allocator_traits<>::value...
#include <numeric>                        // for iota
#include <string>                         // for string
#include <vector>                         // for vector<>::iterator, vector

#include "bio_ik/frame.hpp"       // for Frame, Vector3, Quaternion
#include "bio_ik/robot_info.hpp"  // for RobotInfo

namespace bio_ik {

struct IKEvolution1 : IKBase {
  struct Individual {
    std::vector<double> genes;
    std::vector<double> gradients;
    double extinction;
    double fitness;
  };

  class HeuristicErrorTree {
    size_t variable_count_, tip_count_;
    std::vector<double> table_;
    std::vector<double> chain_lengths_;
    std::vector<std::vector<double>> chain_lengths_2_;

   public:
    HeuristicErrorTree() {}
    HeuristicErrorTree(moveit::core::RobotModelConstPtr robot_model,
                       const std::vector<std::string>& tip_names) {
      tip_count_ = tip_names.size();
      variable_count_ = robot_model->getVariableCount();
      table_.resize(tip_count_ * variable_count_);
      for (size_t tip_index = 0; tip_index < tip_count_; tip_index++) {
        auto& tip_name = tip_names[tip_index];
        for (auto* link_model = robot_model->getLinkModel(tip_name); link_model;
             link_model = link_model->getParentLinkModel()) {
          auto* joint_model = link_model->getParentJointModel();
          size_t v1 = joint_model->getFirstVariableIndex();
          size_t vn = joint_model->getVariableCount();
          for (size_t variable_index = v1; variable_index < v1 + vn;
               variable_index++)
            table_[variable_index * tip_count_ + tip_index] = 1;
        }
      }
      for (size_t variable_index = 0; variable_index < variable_count_;
           variable_index++) {
        double sum = 0;
        for (size_t tip_index = 0; tip_index < tip_count_; tip_index++)
          sum += table_[variable_index * tip_count_ + tip_index];
        if (sum > 0)
          for (size_t tip_index = 0; tip_index < tip_count_; tip_index++)
            table_[variable_index * tip_count_ + tip_index] /= sum;
      }

      chain_lengths_.resize(tip_count_);
      for (size_t tip_index = 0; tip_index < tip_count_; tip_index++) {
        auto& tip_name = tip_names[tip_index];
        double chain_length = 0;
        for (auto* link_model = robot_model->getLinkModel(tip_name); link_model;
             link_model = link_model->getParentLinkModel()) {
          chain_length +=
              Frame(link_model->getJointOriginTransform()).pos.length();
        }
        chain_lengths_[tip_index] = chain_length;
      }

      chain_lengths_2_.resize(tip_count_);
      for (size_t tip_index = 0; tip_index < tip_count_; tip_index++) {
        auto& tip_name = tip_names[tip_index];
        double chain_length = 0;
        chain_lengths_2_[tip_index].resize(variable_count_, 0.0);
        for (auto* link_model = robot_model->getLinkModel(tip_name); link_model;
             link_model = link_model->getParentLinkModel()) {
          auto* joint_model = link_model->getParentJointModel();
          auto vmin = joint_model->getFirstVariableIndex();
          auto vmax = vmin + joint_model->getVariableCount();
          for (size_t vi = vmin; vi < vmax; vi++)
            chain_lengths_2_[tip_index][vi] = chain_length;
          chain_length +=
              Frame(link_model->getJointOriginTransform()).pos.length();
        }
      }
    }
    inline double getInfluence(size_t variable_index, size_t tip_index) const {
      return table_[variable_index * tip_count_ + tip_index];
    }
    inline double getChainLength(size_t tip_index) const {
      return chain_lengths_[tip_index];
    }
    inline double getJointVariableChainLength(size_t tip_index,
                                              size_t variable_index) const {
      return chain_lengths_2_[tip_index][variable_index];
    }
  };

  HeuristicErrorTree heuristicErrorTree_;
  std::vector<double> solution_;
  std::vector<Individual> population_;
  size_t populationSize_;
  size_t eliteCount_;
  std::vector<Individual*> tempPool_;
  std::vector<Individual> tempOffspring_;
  std::vector<double> initialGuess_;

  bool opt_no_wipeout_;

  bool linear_fitness_;

  void setParams(const IKParams& params) {
    opt_no_wipeout_ = params.opt_no_wipeout;
    populationSize_ = static_cast<size_t>(params.population_size);
    eliteCount_ = static_cast<size_t>(params.elite_count);
    linear_fitness_ = params.linear_fitness;
  }

  bool in_final_adjustment_loop_;

  template <class t>
  inline t select(const std::vector<t>& v) {
    // FNPROFILER();
    linear_int_distribution<size_t> d(v.size());
    size_t index = d(rng_);
    return v[index];
  }

  inline double clip(double v, size_t i) { return modelInfo_.clip(v, i); }

  inline double getMutationStrength(size_t i, const Individual& parentA,
                                    const Individual& parentB) {
    double extinction = 0.5 * (parentA.extinction + parentB.extinction);
    double span = modelInfo_.getSpan(i);
    return span * extinction;
  }

  double computeAngularScale(size_t tip_index, const Frame& tip_frame) {
    double angular_scale = sqrt(heuristicErrorTree_.getChainLength(tip_index) *
                                tip_frame.pos.length()) /
                           M_PI;
    return angular_scale;
    // return 1;
    /*double angular_scale = sqrt(heuristicErrorTree_.getChainLength(tip_index)
    * tip_frame.pos.length()) / M_PI;
    //double angular_scale = heuristicErrorTree_.getChainLength(tip_index) *
    (1.0 / M_PI); if(opt_angular_scale_full_circle) angular_scale *= 0.5; return
    angular_scale;*/
  }

  double getHeuristicError(size_t variable_index,
                           [[maybe_unused]] bool balanced) {
    // return 1;

    double heuristic_error = 0;
    // for(int tip_index = 0; tip_index < tipObjectives.size(); tip_index++)
    for (size_t tip_index = 0; tip_index < problem_.goals.size(); tip_index++) {
      double influence =
          heuristicErrorTree_.getInfluence(variable_index, tip_index);
      if (influence == 0) continue;

      // const auto& ta = tipObjectives[tip_index];
      const auto& ta = problem_.goals[tip_index].frame;
      const auto& tb = model_.getTipFrame(tip_index);

      double length = heuristicErrorTree_.getJointVariableChainLength(
          tip_index, variable_index);

      // LOG_ALWAYS("a",
      // heuristicErrorTree_.getJointVariableChainLength(tip_index,
      // variable_index));

      // double length =
      // model_.getJointVariableFrame(variable_index).pos.distance(model_.getTipFrame(tip_index).pos);
      // if(length <= 0.000000001) length = 0;

      // LOG_ALWAYS("b", length);

      if (modelInfo_.isPrismatic(variable_index)) {
        // heuristic_error += ta.pos.distance(tb.pos) * influence;
        // if(length) heuristic_error += ta.rot.angle(tb.rot) * length *
        // influence;

        if (length != 0) {
          heuristic_error += ta.pos.distance(tb.pos) * influence * 0.5;
          heuristic_error += ta.rot.angle(tb.rot) * length * influence * 0.5;
        } else {
          heuristic_error += ta.pos.distance(tb.pos) * influence;
        }
      }

      if (modelInfo_.isRevolute(variable_index)) {
        // if(length) heuristic_error += ta.pos.distance(tb.pos) / length *
        // influence; heuristic_error += ta.rot.angle(tb.rot) * influence;

        if (length != 0) {
          heuristic_error += ta.pos.distance(tb.pos) / length * influence * 0.5;
          heuristic_error += ta.rot.angle(tb.rot) * influence * 0.5;
        } else {
          heuristic_error += ta.rot.angle(tb.rot) * influence;
        }

        // double d = 0.0;
        // if(length) d = std::max(d, ta.pos.distance(tb.pos) / length);
        // d = std::max(d, ta.rot.angle(tb.rot));
        // heuristic_error += d * influence;
      }
    }
    // heuristic_error *= 0.5;
    // LOG_ALWAYS(heuristic_error);
    return heuristic_error;
  }

  bool in_adjustment_2_, in_get_solution_fitness_;

  void reroll(Individual& offspring) {
    FNPROFILER();
    // for(size_t i = 0; i < offspring.genes.size(); i++)
    for (auto i : problem_.active_variables) {
      offspring.genes[i] = random(modelInfo_.getMin(i), modelInfo_.getMax(i));

      offspring.genes[i] =
          mix(offspring.genes[i],
              (modelInfo_.getMin(i) + modelInfo_.getMax(i)) * 0.5,
              random(0.0, 0.1));

      offspring.gradients[i] = 0;
    }
    offspring.fitness = computeFitness(offspring.genes, false);
  }

  double computeFitness(const std::vector<double>& genes, bool balanced) {
    if (linear_fitness_) {
      model_.applyConfiguration(genes);
      double fitness_sum = 0.0;
      for (size_t goal_index = 0; goal_index < problem_.goals.size();
           goal_index++) {
        const auto& ta = problem_.goals[goal_index].frame;
        const auto& tb =
            model_.getTipFrame(problem_.goals[goal_index].tip_index);

        double tdist =
            ta.pos.distance(tb.pos) /
            computeAngularScale(problem_.goals[goal_index].tip_index, ta);
        double rdist = ta.rot.angle(tb.rot);

        fitness_sum +=
            mix(tdist, rdist,
                (balanced || in_final_adjustment_loop_) ? 0.5 : random());
      }
      return fitness_sum;
    } else {
      return IKBase::computeFitness(genes);
    }
  }

  bool checkWipeout() {
    FNPROFILER();
    auto& genes = population_[0].genes;
    // for(size_t i = 0; i < genes.size(); i++)
    for (auto i : problem_.active_variables) {
      double v0 = genes[i];
      double fitness = computeFitness(genes, true);
      double heuristicError = getHeuristicError(i, true);
      // double heuristicError = 0.001;
      genes[i] = modelInfo_.clip(v0 + random(0, heuristicError), i);
      double incFitness = computeFitness(genes, true);
      genes[i] = modelInfo_.clip(v0 - random(0, heuristicError), i);
      double decFitness = computeFitness(genes, true);
      genes[i] = v0;
      if (incFitness < fitness || decFitness < fitness) {
        // LOG("no wipeout");
        return false;
      }
    }
    // LOG("wipeout 1");
    return true;
  }

  void computeExtinctions() {
    double min = population_.front().fitness;
    double max = population_.back().fitness;
    for (size_t i = 0; i < populationSize_; ++i) {
      double grading =
          static_cast<double>(i) / static_cast<double>(populationSize_ - 1);
      population_[i].extinction =
          (population_[i].fitness + min * (grading - 1)) / max;
    }
  }

  bool tryUpdateSolution() {
    FNPROFILER();
    double solutionFitness = computeFitness(solution_, true);
    double candidateFitness = computeFitness(population_[0].genes, true);
    // LOG_VAR(solutionFitness);
    // LOG_VAR(candidateFitness);
    if (candidateFitness < solutionFitness) {
      solution_ = population_[0].genes;
      // solution_ = initialGuess_;
      // for(auto i : problem_.active_variables)
      //    solution_[i] = population_[0].genes[i];
      return true;
    }
    return false;
  }

  double getMutationProbability(const Individual& parentA,
                                const Individual& parentB) {
    double extinction = 0.5 * (parentA.extinction + parentB.extinction);
    double inverse = 1.0 / static_cast<double>(parentA.genes.size());
    return extinction * (1.0 - inverse) + inverse;
  }

  void sortByFitness() {
    FNPROFILER();
    sort(population_.begin(), population_.end(),
         [](const Individual& a, const Individual& b) {
           return a.fitness < b.fitness;
         });
  }

  double bounce(double v, size_t i) {
    double c = clip(v, i);
    v = c - (v - c) * 2;
    // v = c + c - v;
    v = clip(v, i);
    return v;
  }

  void reproduce(Individual& offspring, const Individual& parentA,
                 const Individual& parentB, const Individual& prototype) {
    FNPROFILER();
    for (size_t i = 0; i < offspring.genes.size(); i++)
    // for(auto i : problem_.active_variables)
    {
      offspring.genes[i] = mix(parentA.genes[i], parentB.genes[i], random());
      offspring.genes[i] += parentA.gradients[i] * random();
      offspring.genes[i] += parentB.gradients[i] * random();

      double storage = offspring.genes[i];

      if (random() < getMutationProbability(parentA, parentB))
        offspring.genes[i] +=
            random(-1, 1) * getMutationStrength(i, parentA, parentB);
      // offspring.genes[i] += normal_random() * getMutationStrength(i, parentA,
      // parentB);

      offspring.genes[i] +=
          mix(random() * (0.5 * (parentA.genes[i] + parentB.genes[i]) -
                          offspring.genes[i]),
              random() * (prototype.genes[i] - offspring.genes[i]), random());

      // offspring.genes[i] = clip(offspring.genes[i], i);

      // offspring.genes[i] += fabs(offspring.genes[i] - storage) *
      // offspring.genes[i] - (modelInfo_.getMin(i) + modelInfo_.getMax(i)) *
      // 0.5;

      // offspring.genes[i] = mix(offspring.genes[i], (modelInfo_.getMin(i) +
      // modelInfo_.getMax(i)) * 0.5, random() * 0.1 * fabs(offspring.genes[i] -
      // storage) / modelInfo_.getSpan(i));

      offspring.genes[i] = clip(offspring.genes[i], i);

      // offspring.genes[i] = bounce(offspring.genes[i], i);

      offspring.gradients[i] =
          random() * offspring.gradients[i] + offspring.genes[i] - storage;
    }

    offspring.fitness = computeFitness(offspring.genes, false);
  }

  void exploit(Individual& individual) {
    FNPROFILER();

    double fitness_sum = 0;

    // model_.incrementalBegin(individual.genes);

    for (auto i : problem_.active_variables) {
      double fitness = computeFitness(individual.genes, true);

      double heuristicError = getHeuristicError(i, true);
      double v_0 = individual.genes[i];

      double v_inc = clip(v_0 + random(0, heuristicError), i);
      double v_dec = clip(v_0 - random(0, heuristicError), i);

      individual.genes[i] = v_inc;
      double inc_fitness = computeFitness(individual.genes, true);
      individual.genes[i] = v_dec;
      double dec_fitness = computeFitness(individual.genes, true);

      if (inc_fitness < fitness && inc_fitness <= dec_fitness) {
        individual.genes[i] = v_inc;
        individual.gradients[i] = v_0 * random() + v_inc - v_0;
        fitness_sum += inc_fitness;
      } else if (dec_fitness < fitness && dec_fitness <= inc_fitness) {
        individual.genes[i] = v_dec;
        individual.gradients[i] = v_0 * random() + v_dec - v_0;
        fitness_sum += dec_fitness;
      } else {
        individual.genes[i] = v_0;
        fitness_sum += fitness;
      }
    }

    // model_.incrementalEnd();

    individual.fitness =
        fitness_sum / static_cast<double>(individual.genes.size());
  }

  IKEvolution1(const IKParams& p)
      : IKBase(p),
        populationSize_(12),
        eliteCount_(4),
        in_final_adjustment_loop_(false),
        in_adjustment_2_(false),
        in_get_solution_fitness_(false) {
    setParams(p);
  }

  void init() {
    initialGuess_ = problem_.initial_guess;
    solution_ = initialGuess_;

    population_.resize(populationSize_);

    {
      auto& p = population_[0];
      p.genes = solution_;
      p.gradients.clear();
      p.gradients.resize(p.genes.size(), 0);
      p.fitness = computeFitness(p.genes, false);
    }

    for (size_t i = 1; i < populationSize_; ++i) {
      auto& p = population_[i];
      p.genes = solution_;
      p.gradients.clear();
      p.gradients.resize(p.genes.size(), 0);
      reroll(p);
    }

    sortByFitness();
    computeExtinctions();
  }

  void initialize(const Problem& problem) {
    IKBase::initialize(problem);

    std::vector<std::string> tips;
    for (auto tip_link_index : problem_.tip_link_indices)
      tips.push_back(params_.robot_model->getLinkModelNames()[tip_link_index]);
    heuristicErrorTree_ = HeuristicErrorTree(params_.robot_model, tips);

    init();
  }

  const std::vector<double>& getSolution() const { return solution_; }

  double getSolutionFitness() {
    in_get_solution_fitness_ = true;
    double f = computeFitness(solution_, true);
    in_get_solution_fitness_ = false;
    return f;
  }

  const std::vector<Frame>& getSolutionTipFrames() {
    model_.applyConfiguration(solution_);
    return model_.getTipFrames();
  }

  bool evolve() {
    FNPROFILER();

    auto& offspring = tempOffspring_;
    offspring = population_;

    for (size_t i = 0; i < eliteCount_; ++i) {
      offspring[i] = population_[i];
      exploit(offspring[i]);
    }

    auto& pool = tempPool_;
    pool.resize(populationSize_);
    iota(pool.begin(), pool.end(), &population_[0]);

    for (size_t i = eliteCount_; i < populationSize_; ++i) {
      if (pool.size() > 0) {
        auto& parentA = *select(pool);
        auto& parentB = *select(pool);
        auto& prototype = *select(pool);
        reproduce(offspring[i], parentA, parentB, prototype);
        if (offspring[i].fitness < parentA.fitness)
          pool.erase(remove(pool.begin(), pool.end(), &parentA), pool.end());
        if (offspring[i].fitness < parentB.fitness)
          pool.erase(remove(pool.begin(), pool.end(), &parentB), pool.end());
      } else {
        reroll(offspring[i]);
      }
    }

    population_ = offspring;

    sortByFitness();

    computeExtinctions();

    if (tryUpdateSolution()) return true;
    if (opt_no_wipeout_) return false;
    if (!checkWipeout()) return false;

    init();

    return tryUpdateSolution();
  }

  void step() {
    in_adjustment_2_ = false;
    evolve();
  }

  virtual size_t concurrency() const { return 4; }
};

static IKFactory::Class<IKEvolution1> cIKEvolution1("bio1");

}  // namespace bio_ik
