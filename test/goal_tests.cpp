// #include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <bio_ik/goal_types.hpp>

using namespace bio_ik;

class MyContext : public GoalContext {
public:
  MyContext(const tf2::Vector3 & position = tf2::Vector3(0, 0, 0),
            const tf2::Quaternion & orientation = tf2::Quaternion(0, 0, 0, 1)) :
    fr(position, orientation)
  {
    tip_link_frames_ = &fr;
    goal_link_indices_.push_back(0);
    problem_active_variables_.push_back(0);
    active_variable_positions_ = &var;
    initial_guess_.push_back(0);
    velocity_weights_.push_back(1);
    goal_variable_indices_.push_back(0);
  }

  Frame & getFrame() { return fr; }
  void setActiveVariable(double v) { var = v; }
  void setWeight(double w) { velocity_weights_[0] = w; }

private:
  Frame fr;
  double var = 0;
};

TEST(BioIK, position_goal) {
  // GIVEN a link frame at (0, 0, 0) and a position goal frame at (0,
  // 0, 0).
  PositionGoal goal("", tf2::Vector3(0, 0, 0));
  MyContext context;

  // WHEN we calculate the cost
  // THEN we expect to get 0, since the link frame and goal frame are
  // in the same location.
  EXPECT_EQ(goal.evaluate(context), 0);

  // GIVEN a link frame at (0, 0, 0) and a position goal frame at (1,
  // 0, 0).
  goal.setPosition(tf2::Vector3(1, 0, 0));
  // WHEN we calculate the cost
  // THEN we expect to get 1, which is the square of the magnitude of
  // (1, 0, 0) - (0, 0, 0) = (1, 0, 0).
  EXPECT_EQ(goal.evaluate(context), 1);

  // GIVEN a link frame at (0, 0, 0) and a position goal frame at (1,
  // 1, 0).
  goal.setPosition(tf2::Vector3(1, 1, 0));
  // WHEN we calculate the cost
  // THEN we expect to get 2, which is the square of the magnitude of
  // (1, 1, 0) - (0, 0, 0) = (1, 1, 0).
  EXPECT_EQ(goal.evaluate(context), 2);
}

TEST(BioIK, orientation_goal) {
  // GIVEN a link frame with orientation (0, 0, 0, 1) and a goal frame
  // with orientation (0, 0, 0, 1).
  OrientationGoal goal("", tf2::Quaternion(0, 0, 0, 1));
  MyContext context;
  // WHEN we calculate the cost
  // THEN we expect to get 0, since the link frame and goal frame have
  // the same orientation.
  EXPECT_EQ(goal.evaluate(context), 0);

  // GIVEN a link frame with orientation (0, 0, 0, 1) and a goal frame
  // with orientation (1, 0, 0, 0)
  goal.setOrientation(tf2::Quaternion(1, 0, 0, 0));
  // WHEN we calculate the cost
  // THEN we expect to get 2, which is the square of the magnitude of
  // (1, 0, 0, 0) - (0, 0, 0, 1) = (1, 0, 0, -1).
  EXPECT_EQ(goal.evaluate(context), 2);

  // GIVEN a link frame with orientation (0, 0, 0, 1) and a goal frame
  // with orientation (0, 1, 0, 0)
  goal.setOrientation(tf2::Quaternion(0, 1, 0, 0));
  // WHEN we calculate the cost
  // THEN we expect to get 2, which is the square of the magnitude of
  // (0, 1, 0, 0) - (0, 0, 0, 1) = (0, 1, 0, -1).
  EXPECT_EQ(goal.evaluate(context), 2);

  // GIVEN a link frame with orientation (0, 0, 0, 1) and a goal frame
  // with orientation (0, sqrt(2)/2, sqrt(2)/2, 0)
  goal.setOrientation(tf2::Quaternion(0, 0.5 * std::sqrt(2), 0.5 * std::sqrt(2), 0));
  // WHEN we calculate the cost
  // THEN we expect to get 2, which is the square of the magnitude of
  // (0, sqrt(2)/2, sqrt(2)/2, 0) - (0, 0, 0, 1) = (0, sqrt(2)/2, sqrt(2)/2, -1).
  EXPECT_EQ(goal.evaluate(context), 2);
}

TEST(BioIK, pose_goal) {
  // GIVEN a position goal, and orientation goal, and a pose goal (with rotation scale = 1)
  tf2::Quaternion orientation(0, 0, 0, 1);
  tf2::Vector3 position(0, 0, 0);
  PositionGoal pgoal("", position);
  OrientationGoal ogoal("", orientation);

  double scale = 1;
  PoseGoal goal("", position, orientation);
  goal.setRotationScale(scale);
  MyContext context(position, orientation);

  // WHEN we evaluate the cost
  // THEN we expect that the cost of the pose goal is equal to the
  // cost of the position goal plus the scale squared times the cost
  // of the orientation goal.
  EXPECT_EQ(goal.evaluate(context), 0);
  EXPECT_EQ(goal.evaluate(context), pgoal.evaluate(context) + scale * scale * ogoal.evaluate(context));

  position.setValue(1, 0, 0);
  goal.setPosition(position);
  pgoal.setPosition(position);
  EXPECT_EQ(goal.evaluate(context), 1);
  EXPECT_EQ(goal.evaluate(context), pgoal.evaluate(context) + scale * scale * ogoal.evaluate(context));

  orientation.setValue(1, 0, 0, 0);
  goal.setOrientation(orientation);
  ogoal.setOrientation(orientation);
  EXPECT_EQ(goal.evaluate(context), 3);
  EXPECT_EQ(goal.evaluate(context), pgoal.evaluate(context) + scale * scale * ogoal.evaluate(context));
}

TEST(BioIK, look_at_goal) {
  // GIVEN a link frame with position (0, 0, 0) and orientation (0, 0,
  // 0, 1), and a goal to point the x-axis at (1, 0, 0)
  LookAtGoal goal("", tf2::Vector3(1, 0, 0), tf2::Vector3(1, 0, 0));
  MyContext context;

  // WHEN we evaluate the cost.
  // THEN we expect to get 0, since the point (1, 0, 0) lies on the
  // local x-axis.
  EXPECT_EQ(goal.evaluate(context), 0);

  // GIVEN the same link frame, and a goal point at (1, 1, 0)
  goal.setTarget(tf2::Vector3(1, 1, 0));
  // WHEN we calculate the cost
  // THEN we expect to get 0.58579.
  EXPECT_NEAR(goal.evaluate(context), 0.58579, 1e-3);

  // GIVEN the same link frame, and a goal point at (1, 1, 1)
  goal.setTarget(tf2::Vector3(1, 1, 1));
  // WHEN we calculate the cost
  // THEN we expect to get 0.84530.
  EXPECT_NEAR(goal.evaluate(context), 0.84530, 1e-3);

  // GIVEN a link frame with position (0, 1, 0) and orientation (0, 0,
  // 0, 1) and a goal to point the x-axis at (1, 0, 0).
  context.getFrame().setPosition(tf2::Vector3(0, 1, 0));
  goal.setTarget(tf2::Vector3(1, 0, 0));
  // WHEN we calculate the cost
  // THEN we expect to get 0.58579.
  EXPECT_NEAR(goal.evaluate(context), 0.58579, 1e-3);
}

TEST(BioIK, max_distance_goal) {
  // GIVEN a link frame at (0, 0, 0), a goal frame at (0, 0, 0), and a
  // max. distance of 1
  MaxDistanceGoal goal("", tf2::Vector3(0, 0, 0), 1);
  MyContext context;

  // WHEN we evaluate the cost
  // THEN we expect to get zero, since the distance between the two
  // frames is less than the max. distance.
  EXPECT_EQ(goal.evaluate(context), 0);

  // GIVEN a goal frame at (0.5, 0, 0)
  goal.setTarget(tf2::Vector3(0.5, 0, 0));
  // WHEN we evaluate the cost
  // THEN we expect to get zero, since the distance between the two
  // frames is less than the max. distance.
  EXPECT_EQ(goal.evaluate(context), 0);

  // GIVEN a goal frame at (1, 0, 0)
  goal.setTarget(tf2::Vector3(1, 0, 0));
  // WHEN we evaluate the cost
  // THEN we expect to get zero, since the distance between the two
  // frames is equal to the max. distance.
  EXPECT_EQ(goal.evaluate(context), 0);

  // GIVEN a goal frame at (1.5, 0, 0)
  goal.setTarget(tf2::Vector3(1.5, 0, 0));
  // WHEN we evaluate the cost
  // THEN we expect to get 0.25, which is the square the quantity
  // (distance between goal frames) - (max. distance).
  EXPECT_NEAR(goal.evaluate(context), 0.25, 1e-3);
}

TEST(BioIK, min_distance_goal) {
  // GIVEN a link frame at (0, 0, 0), a goal frame at (0, 0, 0), and a
  // min. distance of 1
  MinDistanceGoal goal("", tf2::Vector3(0, 0, 0), 1);
  MyContext context;

  // WHEN we evaluate the cost
  // THEN we expect to get 1, which is the square of the quantity
  // (distance between frames) - (min. distance).
  EXPECT_EQ(goal.evaluate(context), 1);

  // GIVEN a goal frame at (0.5, 0, 0)
  goal.setTarget(tf2::Vector3(0.5, 0, 0));
  // WHEN we evaluate the cost
  // THEN we expect to get 0.25.
  EXPECT_NEAR(goal.evaluate(context), 0.25, 1e-3);

  // GIVEN a goal frame at (1, 0, 0)
  goal.setTarget(tf2::Vector3(1, 0, 0));
  // WHEN we evaluate the cost
  // THEN we expect to get zero, since the distance between the two
  // frames is equal to the min. distance.
  EXPECT_EQ(goal.evaluate(context), 0);

  // GIVEN a goal frame at (1.5, 0, 0)
  goal.setTarget(tf2::Vector3(1.5, 0, 0));
  // WHEN we evaluate the cost
  // THEN we expect to get 0, since the distance between the frames is
  // greater than the min. distance.
  EXPECT_EQ(goal.evaluate(context), 0);
}

TEST(BioIK, regularization_goal) {
  // GIVEN an a joint position of 0, and an initial guess of 0.
  RegularizationGoal goal;
  MyContext context;

  // WHEN we evaluate the cost
  // THEN we expect to get 0, since the current joint position matches
  // the initial guess.
  EXPECT_EQ(goal.evaluate(context), 0);

  for (double v = 0; v <= 1; v += 0.1) {
    // GIVEN an arbitrary joint position, and an initial guess of 0.
    context.setActiveVariable(v);
    // WHEN we evaluate the cost
    // THEN we expect to get the square of the joint position.
    EXPECT_NEAR(goal.evaluate(context), v * v, 1e-3);
  }
}

TEST(BioIK, minimial_displacement_goal) {
  // GIVEN a RegularizationGoal and a MinimalDisplacementGoal, and a weight of 1
  RegularizationGoal rgoal;
  MinimalDisplacementGoal mgoal;
  MyContext context;

  // WHEN we compare the costs of the RegularizationGoal and MinimalDisplacementGoal
  // THEN we expect they are the same
  EXPECT_EQ(rgoal.evaluate(context), mgoal.evaluate(context));

  for (double v = 0; v <= 1; v += 0.1) {
    context.setActiveVariable(v);
    EXPECT_EQ(rgoal.evaluate(context), mgoal.evaluate(context));
  }

  // GIVEN a RegularizationGoal and a MinimalDisplacementGoal, and a weight of 0.5
  context.setWeight(0.5);

  // WHEN we compare the costs of the RegularizationGoal and MinimalDisplacementGoal
  // THEN we expect the cost of the MinimalDisplacementGoal to be
  // smaller by a factor of 0.5 * 0.5 = 0.25.
  EXPECT_EQ(0.25 * rgoal.evaluate(context), mgoal.evaluate(context));

  for (double v = 0; v <= 1; v += 0.1) {
    context.setActiveVariable(v);
    EXPECT_EQ(0.25 * rgoal.evaluate(context), mgoal.evaluate(context));
  }
}

TEST(BioIK, joint_variable_goal) {
  // GIVEN a joint value of 0, and a JointVariableGoal of 0
  JointVariableGoal goal("", 0);
  MyContext context;
  context.setActiveVariable(0);

  // WHEN we evalute the cost
  // THEN we expect to get 0, since the target and actual joint values
  // are the same.
  EXPECT_EQ(goal.evaluate(context), 0);

  // GIVEN a joint value of 0, and a target value of 0.5
  goal.setVariablePosition(0.5);
  // WHEN we evaluate the cost
  // THEN we expect to get (0.5 - 0) ^ 2 = 0.25
  EXPECT_NEAR(goal.evaluate(context), 0.25, 1e-3);

  // GIVEN a joint value of 0.5, and a target value of 0
  context.setActiveVariable(0.5);
  goal.setVariablePosition(0);
  // WHEN we evaluate the cost
  // THEN we expect to get (0 - 0.5) ^ 2 = 0.25
  EXPECT_NEAR(goal.evaluate(context), 0.25, 1e-3);
}

TEST(BioIK, joint_function_goal) {
  // GIVEN a JointFunctionGoal that sets all temp values to 1, and an
  // initial joint value of 0.
  JointFunctionGoal goal({""}, [](std::vector<double>& v) { std::fill(v.begin(), v.end(), 1); });
  MyContext context;

  // WHEN we evaluate the cost
  // THEN we expect to get 1
  EXPECT_EQ(goal.evaluate(context), 1);

  // GIVEN an initial joint value of -1.
  context.setActiveVariable(-1);

  // WHEN we evaluate the cost
  // THEN we expect to get (-1 - 1) ^ 2 = 4.
  EXPECT_EQ(goal.evaluate(context), 4);
}

TEST(BioIK, link_function_goal) {
  // GIVEN a LinkFunctionGoal that returns the square of the distance
  // from the origin, and an initial position of (1, 1, 1).
  LinkFunctionGoal goal("", [](const tf2::Vector3& v, [[maybe_unused]] const tf2::Quaternion& q) { return v.length2(); });
  MyContext context(tf2::Vector3(1, 1, 1));

  // WHEN we evaluate the cost
  // THEN we expect to get 1^2 + 1^2 + 1^2 = 3.
  EXPECT_EQ(goal.evaluate(context), 3);

  // GIVEN a LinkFunctionGoal that returns the square of the distance
  // from the point (1, 1, 1), and the same link positon.
  goal.setLinkFunction([](const tf2::Vector3& v, [[maybe_unused]] const tf2::Quaternion& q) { return v.distance2(tf2::Vector3(1, 1, 1)); });

  // WHEN we evaluate the cost
  // THEN we expect to get 0.
  EXPECT_EQ(goal.evaluate(context), 0);
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
