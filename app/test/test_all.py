from app.labels_collector.labels_collector import *
from app.lane_detector.lane_detector import *
from app.dates_recognition.digits_extractor import *
from app.dates_recognition.dates_searcher import *
from app.classification_service.classification_service import *
from test_data_loader import *
from app.utils.param_utils import create_params_combs_dicts



# Parameters for Finding Dates
PARAM_GRID = {
    'thresh': [80],  # 20],
    'blue_thresh': [False],  # True],
    'binary_roi': [True],  # False],
    'separate_c': [False],  # True],
    'blur': [0],  # 3, 5],
    'brightness_inc': [35, 0],
    'contrast_inc': [0],  # 2],
    'opening_shape': [disk(3), None],
    'closing_shape': [None, disk(2)],
    'dilation_size': [0, 2],  # 3],
    'erosion_size': [0, 3],  # 4],
    'should_deskew': [True],  # False]
}


def test_all():
    ##################
    # Load test images
    ##################

    imgs_nov = load_nov_2016_images()
    imgs_april = load_april_2016_images()
    all_images = imgs_nov + imgs_april


    #################
    # Extracting Lanes
    #################

    # Grab Albumin roi
    alb = imgs_nov[0][307:398, 460:507]
    lanes_per_gel, all_lanes = get_the_lanes(all_images, alb)

    print('Lanes per gel len: ', len(lanes_per_gel))
    print('All lanes len: ', len(all_lanes))
    print 'ORIG_IMAGES len: ', len(ORIG_IMAGES)

    # Gold std lanes
    gld = all_lanes[0][0]
    # gld_bad = all_lanes[279][0]
    gld_bad = all_lanes[179][0]
    dist_labels = good_bad_euclid(gld, gld_bad, all_lanes)
    # Use min(good_dist, bad_dist)
    good_lanes_per_gel, bad_lanes_per_gel = group_lanes_per_date(all_lanes, lanes_per_gel, dist_labels, 0)
    print len(good_lanes_per_gel), len(good_lanes_per_gel[0])


    ###################
    # Extracting Dates
    ###################

    params_combs_dicts = create_params_combs_dicts(PARAM_GRID)

    found_boolv = []
    found_dates = []
    for im in ORIG_IMAGES:
        for params in params_combs_dicts:
            im_c, rois, date_possibs, probs = extract_numbers(im, **params)
            dates = find_dates(date_possibs)
            if len(dates):
                print "Found dates: ", dates
                found_dates.append(dates[0]) # Todo - just take first found date for now...
                found_boolv.append(1)
                break
        found_boolv.append(0)

    print("Found boolv: ", found_boolv)


    ##################
    # Gathering Labels
    ##################

    labels = [retreive_labels(datetime.strptime(fd, '%Y-%m-%d'), datetime.strptime(fd, '%Y-%m-%d')) for fd in found_dates]

    good_lanes_per_gel_filt = [x for i, x in enumerate(good_lanes_per_gel) if found_boolv[i] == 1]
    X = [zip(calc_lane_means([z[0] for z in x]), [z[1] for z in x]) for x in good_lanes_per_gel_filt]

    print labels, found_dates, len(X)

    ys = [build_labels(labels[i], found_dates[i], x) for i, x in enumerate(X)]
    y = [y for yx in ys for y in yx]
    X_flat = [z[0] for x in X for z in x]
    print len(X), len(X_flat), len(y)

    print len(imgs_nov), len(labels)


    ##################
    # Do the classification
    ##################

    auto_classify(X_flat, y)


if __name__ == '__main__':
    test_all()