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
  }

  Frame & getFrame() { return fr; }

private:
  Frame fr;
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

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
