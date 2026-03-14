# Basketball Shooting Drills Knowledge Base

Source reference: https://www.basketballforcoaches.com/basketball-shooting-drills/

Purpose: Retrieval dataset for PureArc coaching agents.

How to use in RAG:
- Retrieve by issue tags (for example: `slow_shot_release`, `poor_balance`, `pressure_shooting`).
- Return 1-2 drills per shot based on top detected issues.
- Keep drill title, purpose, and instructions in response.

---

## Drill: Hand-Off Shooting Drill
id: drill_handoff_shooting
purpose: Practice shooting after receiving a hand-off while moving.
fixes:
- slow_shot_release
- poor_rhythm_shooting
- bad_footwork_after_handoffs
instructions:
1. Player starts near the top of the key.
2. Teammate provides a hand-off.
3. Shooter takes 1-2 steps and shoots.
4. Rotate after shot.
tags:
- catch_and_shoot
- footwork
- rhythm
- game_speed

---

## Drill: 23 Cones Shooting Drill
id: drill_23_cones
purpose: Competitive shooting drill that creates pressure situations.
fixes:
- missing_shots_under_pressure
- inconsistent_focus
- game_pressure_shooting
instructions:
1. Place 23 cones at the opposite baseline.
2. Split players into two teams.
3. Players shoot and sprint to the other end after a make.
4. A made shot earns a cone for the team.
tags:
- pressure_shooting
- competition
- focus

---

## Drill: Pressure Jump Shots
id: drill_pressure_jump_shots
purpose: Practice making jump shots while under mental pressure.
fixes:
- nervous_shooting
- inconsistent_jump_shots
- poor_focus_during_games
instructions:
1. Four lines form at the elbows.
2. Players must make two shots from each elbow.
3. Missed shots require repeating the attempt.
tags:
- jump_shot
- pressure
- focus

---

## Drill: Speed Shooting Drill
id: drill_speed_shooting
purpose: Improve shooting accuracy while fatigued.
fixes:
- bad_balance_after_sprinting
- poor_conditioning_shooting
- rushed_shot_mechanics
instructions:
1. Players sprint down the court with the ball.
2. Stop and shoot.
3. Rebound and sprint back for another shot.
tags:
- conditioning
- fatigue_shooting
- balance
- tempo

---

## Drill: Off-the-Dribble Form Shooting
id: drill_off_dribble_form
purpose: Develop correct footwork when shooting off the dribble.
fixes:
- bad_shooting_balance
- poor_1_2_step_mechanics
- inconsistent_pull_up_jumpers
instructions:
1. Players perform pump fakes.
2. Use 1-2 step or hop footwork.
3. Shoot off the dribble.
tags:
- off_dribble
- footwork
- pull_up
- balance

---

## Drill: Weave Layups
id: drill_weave_layups
purpose: Practice fast break layups and passing.
fixes:
- missed_layups_at_speed
- poor_fastbreak_finishing
instructions:
1. Three players weave from half court.
2. Wing finishes with layup.
3. Rebound and reset.
tags:
- layups
- fastbreak
- finishing

---

## Drill: Screen Shooting
id: drill_screen_shooting
purpose: Practice scoring after using an off-ball screen.
fixes:
- poor_movement_shooting
- bad_positioning_after_screens
instructions:
1. Player cuts down the lane.
2. Uses a screen.
3. Receives pass and shoots.
tags:
- off_ball
- screens
- movement_shooting

---

## Drill: 30 and 1 Shooting Drill
id: drill_30_and_1
purpose: Shooting from multiple spots under competition.
fixes:
- inconsistent_shooting_spots
- poor_long_range_confidence
instructions:
1. Teams must make:
   - 10 shots from block
   - 10 shots from elbow
   - 10 shots from 3-point line
2. Finish with one long-range shot.
tags:
- spot_shooting
- range
- competition

---

## Drill: Chase Down Layups
id: drill_chase_down_layups
purpose: Practice finishing while being chased by defenders.
fixes:
- poor_layup_finishing_under_pressure
instructions:
1. Player attacks the basket.
2. Defender chases from behind.
3. Finish quickly.
tags:
- layups
- pressure
- finishing

---

## Drill: Partner Shooting
id: drill_partner_shooting
purpose: High-repetition shooting with a rebounder.
fixes:
- low_shot_volume
- inconsistent_mechanics
instructions:
1. One player shoots.
2. Partner rebounds and passes.
3. Shooter backpedals and repeats.
tags:
- repetition
- rhythm
- conditioning
- volume

---

## Drill: 5 Spot Variety Shooting
id: drill_5_spot_variety
purpose: Practice multiple types of shots from different spots.
fixes:
- lack_of_shooting_versatility
instructions:
1. Five spots around the court.
2. Players take four different shots at each spot.
tags:
- spot_shooting
- versatility
- fundamentals

---

## Drill: Drive and Kick Shooting
id: drill_drive_kick
purpose: Practice shooting after a teammate drives and passes.
fixes:
- slow_catch_and_shoot
- poor_spacing_shooting
instructions:
1. Player drives toward the basket.
2. Pass to open shooter.
3. Shooter takes catch-and-shoot jumper.
tags:
- catch_and_shoot
- team_offense
- spacing

---

## Drill: Titan Shooting
id: drill_titan_shooting
purpose: Team shooting with conditioning.
fixes:
- fatigue_shooting_mechanics
- team_shooting_pace
instructions:
1. Three lines at free throw area.
2. Shoot, rebound, pass.
3. Run to another line before next shot.
tags:
- conditioning
- repetition
- shooting_volume
- pace

---

## Drill: Rainbow Shooting
id: drill_rainbow_shooting
purpose: Warm-up drill with shooting from many spots.
fixes:
- cold_shooting_starts
- poor_shooting_rhythm
instructions:
1. Two lines with basketballs.
2. Players rotate shooting from multiple spots.
tags:
- warmup
- repetition
- rhythm

---

## Drill: 31 Shooting Drill
id: drill_31_shooting
purpose: Competitive shooting from multiple ranges.
fixes:
- inconsistent_scoring_from_different_distances
instructions:
1. Each player takes:
   - one 3-point shot
   - one mid-range shot
   - one inside shot
2. Points accumulate until 31.
tags:
- competition
- range
- scoring

---

## Issue-to-Drill Tag Mapping (for backend deterministic selection)

flat_arc:
- arc_control
- rhythm
- fundamentals

low_release:
- form
- fundamentals
- repetition

rushed_shot:
- rhythm
- footwork
- tempo

arm_shooting:
- conditioning
- balance
- form

fading:
- balance
- fundamentals
- footwork

slow_load:
- game_speed
- repetition
- catch_and_shoot

asymmetric_arc:
- rhythm
- fundamentals
- form

Note: This section is for developer logic and can also be included in RAG retrieval context.
