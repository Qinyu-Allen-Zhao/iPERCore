import joblib
import numpy as np

from iPERCore.models import ModelsFactory
from iPERCore.tools.utils.signals.smooth import temporal_smooth_smpls
from iPERCore.tools.utils.filesio.persistence import clear_dir
from iPERCore.tools.utils.multimedia.video import fuse_src_ref_multi_outputs
from iPERCore.services.preprocess import preprocess
from iPERCore.services.personalization import personalize
from iPERCore.services.options.process_info import ProcessInfo
from iPERCore.services.options.meta_info import MetaImitateOutput
from iPERCore.services.base_runner import (
    get_src_info_for_inference,
    add_hands_params_to_smpl,
    add_special_effect,
    add_bullet_time_effect
)


def call_imitator_inference(opt, imitator, out_dir, prefix,
                            ref_smpls, visualizer, use_selected_f2pts=False):
    """

    Args:
        opt:
        imitator:
        out_dir:
        ref_smpls:
        visualizer:
        use_selected_f2pts:

    Returns:
        outputs (List[Tuple[str]]):
    """

    # add hands parameters to smpl
    ref_smpls = add_hands_params_to_smpl(ref_smpls, imitator.body_rec.np_hands_mean)

    # run imitator's inference function
    outputs = imitator.inference(tgt_smpls=ref_smpls, cam_strategy="no",
                                 output_dir=out_dir, prefix=prefix, visualizer=visualizer,
                                 verbose=True, use_selected_f2pts=use_selected_f2pts)
    outputs = list(zip(outputs))

    results_dict = {
        "outputs": outputs
    }

    return results_dict


def imitate(opt):
    """

    Args:
        opt:

    Returns:
        all_meta_outputs (list of MetaOutput):

    """

    print("Step 3: running imitator.")

    if opt.ip:
        from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer
        visualizer = VisdomVisualizer(env=opt.model_id, ip=opt.ip, port=opt.port)
    else:
        visualizer = None

    # set imitator
    imitator = ModelsFactory.get_by_name("imitator", opt)

    meta_src_proc = opt.meta_data["meta_src"]

    # COCO smpls
    coco_info = joblib.load('./results/synthesis/ref_info.pkl')

    all_meta_outputs = []
    for i, meta_src in enumerate(meta_src_proc):
        """
        meta_input:
                path: /p300/tpami/neuralAvatar/sources/fange_1/fange_1_ns=2
                bg_path: /p300/tpami/neuralAvatar/sources/fange_1/IMG_7225.JPG
                name: fange_1
        primitives_dir: ../tests/debug/primitives/fange_1
        processed_dir: ../tests/debug/primitives/fange_1/processed
        vid_info_path: ../tests/debug/primitives/fange_1/processed/vid_info.pkl
        """
        src_proc_info = ProcessInfo(meta_src)
        src_proc_info.deserialize()

        src_info = src_proc_info.convert_to_src_info(num_source=opt.num_source)
        src_info_for_inference = get_src_info_for_inference(opt, src_info)

        # source setup
        imitator.source_setup(
            src_path=src_info_for_inference["paths"],
            src_smpl=src_info_for_inference["smpls"],
            masks=src_info_for_inference["masks"],
            bg_img=src_info_for_inference["bg"],
            offsets=src_info_for_inference["offsets"],
            links_ids=src_info_for_inference["links"],
            visualizer=visualizer
        )

        """
        meta_input:
            path: /p300/tpami/neuralAvatar/references/videos/bantangzhuyi_1.mp4
            bg_path: 
            name: bantangzhuyi_1
            audio: /p300/tpami/neuralAvatar/references/videos/bantangzhuyi_1.mp3
            fps: 30.02
            pose_fc: 400.0
            cam_fc: 150.0
        primitives_dir: ../tests/debug/primitives/bantangzhuyi_1
        processed_dir: ../tests/debug/primitives/bantangzhuyi_1/processed
        vid_info_path: ../tests/debug/primitives/bantangzhuyi_1/processed/vid_info.pkl
        """

        choices = np.random.choice(range(len(coco_info['smpls'])), 50, replace=False)
        ref_info = {
            "smpls": coco_info['smpls'][choices]
        }
        joblib.dump(ref_info, "/content/drive/MyDrive/datasets/syn_poses/sub%d.pkl" % opt.sub_id)
        out_dir = "/content/drive/MyDrive/datasets/synthesis_dataset/"
        prefix = 'sub%d_' % opt.sub_id
        _ = call_imitator_inference(
            opt, imitator, out_dir, prefix,
            ref_smpls=ref_info["smpls"],
            visualizer=visualizer
        )

    print("Step 3: running imitator done.")
    return all_meta_outputs


def run_imitator(opt):
    # 1. prepreocess
    successful = preprocess(opt)

    if successful:
        # 2. personalization
        personalize(opt)
        # 3. imitate
        all_meta_outputs = imitate(opt)
    else:
        all_meta_outputs = []

    return all_meta_outputs


if __name__ == "__main__":
    from iPERCore.services.options.options_inference import InferenceOptions

    OPT = InferenceOptions().parse()
    run_imitator(opt=OPT)
