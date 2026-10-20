## E1 / E4 navigation
|            |   runs |   success_rate |   person_contacts |   person_contacts_robot_moving |   pet_contacts |   pet_contacts_robot_moving |   furniture_contacts |   min_person_dist_m |   median_min_person_dist_m |   mean_time_s(success) |   falls |
|:-----------|-------:|---------------:|------------------:|-------------------------------:|---------------:|----------------------------:|---------------------:|--------------------:|---------------------------:|-----------------------:|--------:|
| (False, 2) |    100 |           0.99 |                27 |                             26 |              0 |                           0 |                    0 |              -0.444 |                      0.04  |                   39.6 |       0 |
| (True, 1)  |    100 |           0.99 |                 0 |                              0 |              0 |                           0 |                    0 |             inf     |                    inf     |                   26   |       0 |
| (True, 2)  |    100 |           0.94 |                16 |                              7 |              0 |                           0 |                    0 |              -0.226 |                      0.047 |                   46.2 |       0 |
| (True, 3)  |    100 |           0.8  |                13 |                             11 |              7 |                           6 |                    0 |              -0.325 |                      0.048 |                   54.3 |       0 |

## E2 manipulation (e2_manip_expert.csv)
|                     |   count |   mean |
|:--------------------|--------:|-------:|
| ('expert', 'pick')  |      50 |      1 |
| ('expert', 'place') |      50 |      1 |

## E3 full task (e3_full_level2_expert.csv)
success 26/30 = 87%; person contacts 7 (robot moving 7)
| failed_at             |   count |
|:----------------------|--------:|
| MANIPULATE(place_cup) |       3 |
| NAVIGATE_TO(sink)     |       1 |
