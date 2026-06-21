Step6 replay-based control evaluation using NEW Step5 CBF predictions
=================================================================

Interpretation:
- Results estimate resource-saving potential under historical replay conditions.
- Reduced actions were not actually implemented in the greenhouse.
- The outputs should not be interpreted as validated closed-loop control savings.
- The replay setting cannot fully simulate counterfactual greenhouse dynamics.

Inputs:
- raw_csv: D:\JetBrains\Python\PyCharmProjects\GreenHouse·TomatoExperimentTriggerImprovement\step0_clean_build\step0_results\data_trigger_mainmodel_raw.csv
- predictions_csv: D:\JetBrains\Python\PyCharmProjects\GreenHouse·TomatoExperimentTriggerImprovement\step5_cbf\step5_cbf_results_leafT_review_forecasting_stats\cbf_test_predictions_all.csv
- fold_metrics_csv: D:\JetBrains\Python\PyCharmProjects\GreenHouse·TomatoExperimentTriggerImprovement\step5_cbf\step5_cbf_results_leafT_review_forecasting_stats\cbf_forecasting_full_results_by_fold.csv
- best_by_horizon_csv: D:\JetBrains\Python\PyCharmProjects\GreenHouse·TomatoExperimentTriggerImprovement\step5_cbf\step5_cbf_results_leafT_review_forecasting_stats\cbf_forecasting_best_by_horizon.csv

Selected prediction configurations:
 horizon_minutes  selected_feature_set selected_feature_display_name selected_model  n_prediction_rows                           selection_reason
              10 TwoStage_CBF_AirT_PAR                 Two-stage CBF       PatchTST               5040 exact selected Two-stage CBF configuration
              30 TwoStage_CBF_AirT_PAR                 Two-stage CBF       PatchTST               5040 exact selected Two-stage CBF configuration
              60 TwoStage_CBF_AirT_PAR                 Two-stage CBF       PatchTST               5040 exact selected Two-stage CBF configuration
             120 TwoStage_CBF_AirT_PAR                 Two-stage CBF       PatchTST               5040 exact selected Two-stage CBF configuration

Default replay parameters:
- auto_Tlow_quantile: 0.1
- default_margin: 0.25
- default_reduce: 1.0
- default_k: 0.25
- violation_reference: T_low
