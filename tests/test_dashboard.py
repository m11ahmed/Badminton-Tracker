"""Dashboard behavior checks, including actual completed sample results."""
from __future__ import annotations
import io
import json
from pathlib import Path
import sys

import pytest
from PIL import Image
from badminton_tracker import dashboard

ROOT = Path(__file__).resolve().parents[1]


def test_result_paths_reject_escape_and_root(tmp_path):
    with pytest.raises(ValueError): dashboard.contained_path(tmp_path,tmp_path)
    with pytest.raises(ValueError): dashboard.contained_path(tmp_path,tmp_path.parent/'outside','results')
    assert dashboard.contained_path(tmp_path,'results/run','results') == tmp_path/'results'/'run'


def test_completed_m5_runs_sort_first_and_incomplete_skipped(tmp_path):
    for name, milestone, status in [('m5-old','M5','complete'),('m3-new','M3','complete'),('partial','M5','running')]:
        folder=tmp_path/'results'/name;folder.mkdir(parents=True)
        (folder/'summary.json').write_text(json.dumps({'status':status,'provenance':{'milestone':milestone}}))
    assert [path.name for path in dashboard.list_runs(tmp_path)] == ['m5-old','m3-new']


def test_position_rows_keep_missing_unavailable_and_real_coordinates():
    frames=[{'frame_index':2,'timestamp_s':.04,'players':[{'track_id':1,'court_position':{'x_m':3.1,'y_m':8.4}},{'track_id':2,'court_position':None}]}]
    assert dashboard.position_rows(frames)==[{'frame_index':2,'timestamp_s':.04,'track_id':1,'x_m':3.1,'y_m':8.4}]
    image=Image.open(io.BytesIO(dashboard.court_image(dashboard.position_rows(frames),frame_index=2)))
    assert image.size==(360,720)


def test_analysis_is_explicit_argument_list_and_confined(tmp_path):
    (tmp_path/'data').mkdir(); video=tmp_path/'data'/'video.mp4';video.touch()
    python=tmp_path/'.venv'/'Scripts'/'python.exe';python.parent.mkdir(parents=True);python.touch()
    calibration=tmp_path/'configs'/'court.json';calibration.parent.mkdir();calibration.touch()
    command, output=dashboard.analysis_command(tmp_path,video,'doubles',calibration,max_frames=50)
    assert command[command.index('--players-mode')+1]=='doubles'
    assert command[command.index('--milestone')+1]=='m5'
    assert command[command.index('--max-frames')+1]=='50'
    assert '--auto-court' not in command
    assert command[command.index('--pose-nms-iou')+1]=='0.7'
    assert output.parent==tmp_path/'results'
    with pytest.raises(ValueError):dashboard.analysis_command(tmp_path,video,'singles',None)
    with pytest.raises(ValueError):dashboard.analysis_command(tmp_path,tmp_path.parent/'outside.mp4','singles',calibration)


def test_pose_nms_preset_only_applies_to_prepared_singles_example():
    example=ROOT/'data'/'lee_axelsen_rally.mp4'
    other=ROOT/'data'/'other_camera.mp4'
    assert dashboard.prepared_pose_nms_iou(example,'singles')==.5
    assert dashboard.prepared_pose_nms_iou(example,'doubles')==.7
    assert dashboard.prepared_pose_nms_iou(other,'singles')==.7


def test_real_m3_review_saves_new_files_and_binds_source(tmp_path):
    source=ROOT/'results'/'m3-sample';run=tmp_path/'results'/'same-sample';run.mkdir(parents=True)
    summary=dashboard.read_json(source/'summary.json');(run/'summary.json').write_text(json.dumps(summary))
    frames=dashboard.frames_of(dashboard.read_json(source/'players.json'))
    review={'schema_version':1,'identity_map':{'1':{'player_label':'Player A','side':'near'}},'events':[],'rallies':[],'recovery_targets':[]}
    first=dashboard.save_review(tmp_path,run,review,frames);second=dashboard.save_review(tmp_path,run,review,frames)
    assert first!=second
    assert dashboard.read_json(first)['source_video_sha256']==summary['provenance']['input_video_sha256']
    bad={**review,'source_video_sha256':'0'*64}
    with pytest.raises(ValueError,match='different source'): dashboard.save_review(tmp_path,run,bad,frames)


def test_manual_player_region_checks_geometry(tmp_path):
    (tmp_path/'configs').mkdir()
    path=dashboard.write_roi(tmp_path,[[.25,.30],[.75,.30],[.9,1],[.1,1]])
    assert dashboard.read_json(path)['points'][2]==[.9,1]
    with pytest.raises(ValueError):dashboard.write_roi(tmp_path,[[.25,.30],[.75,.30],[.9,1],[1.2,1]])


def make_app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()


def choose_run(app, name):
    app.selectbox(key='selected_run').set_value(ROOT/'results'/name).run()
    assert not app.exception
    return app


def press_form(app,label):
    next(button for button in app.button if button.label==label).click().run()
    assert not app.exception
    return app


def test_apptest_real_m3_and_partial_doubles_review():
    app=choose_run(make_app(),'m3-sample')
    assert any(metric.label=='Frames reviewed' and metric.value=='81' for metric in app.metric)
    assert any('M5' in notice.value for notice in app.info)
    app=choose_run(app,'m2-rear-doubles-check')
    assert any(metric.label=='Analyzed duration' and metric.value=='3.00 s' for metric in app.metric)
    assert any('players than expected' in warning.value for warning in app.warning)
    assert any('after M3' in notice.value for notice in app.info)


def test_apptest_real_m5_sample_metrics_and_shuttle_limit():
    if not (ROOT/'results'/'m5-sample'/'summary.json').is_file():pytest.skip('M5 sample is not available')
    app=choose_run(make_app(),'m5-sample')
    assert any(metric.label=='Distance retained' and metric.value=='4.11 m' for metric in app.metric)
    assert any(metric.label=='Physical shuttle speed' and metric.value=='Unavailable' for metric in app.metric)
    assert any('3D information' in notice.value for notice in app.info)
    assert any('No rally boundaries reviewed' in notice.value for notice in app.info)
    rally_table=next(table.value for table in app.dataframe if 'Rally' in table.value.columns)
    assert list(rally_table.columns)==['Rally','Start frame','End frame','Duration (s)','Outcome','Notes']
    assert rally_table.iloc[0]['Rally']=='Clip span'
    assert 'boundary_source' not in rally_table.columns and 'players' not in rally_table.columns


def test_apptest_exposes_large_shuttle_jump_review_warning():
    run=ROOT/'results'/'m5_lee_rally_20261004_004704_911'
    if not (run/'summary.json').is_file():
        pytest.skip('Verified full rally result is not available')
    app=choose_run(make_app(),run.name)
    assert any('does not distinguish a fast shuttle from a false detection' in warning.value for warning in app.warning)


def test_apptest_labels_reject_simultaneous_same_athlete_and_bad_contact():
    app=choose_run(make_app(),'m3-sample')
    app.text_input(key='identity_label').input('Player A')
    app.selectbox(key='identity_side').select('near')
    press_form(app,'Add or update player label')
    assert app.session_state['review_m3-sample']['identity_map']['1']['player_label']=='Player A'
    app.selectbox(key='identity_fragment').select(2)
    app.text_input(key='identity_label').input('Player A')
    press_form(app,'Add or update player label')
    assert app.error
    assert '2' not in app.session_state['review_m3-sample']['identity_map']
    app.text_input(key='contact_player').input('Unknown athlete')
    press_form(app,'Add annotation')
    assert app.error
    assert app.session_state['review_m3-sample']['events']==[]


def test_apptest_valid_contacts_targets_and_saved_review(monkeypatch,tmp_path):
    app=choose_run(make_app(),'m3-sample')
    app.text_input(key='identity_label').input('Player A')
    app.selectbox(key='identity_side').select('near')
    press_form(app,'Add or update player label')
    app.text_input(key='contact_player').input('Player A')
    app.text_input(key='contact_note').input('Late rear court return')
    app.number_input(key='contact_frame').set_value(20)
    press_form(app,'Add annotation')
    app.text_input(key='target_player').input('Player A')
    app.number_input(key='target_contact').set_value(20)
    press_form(app,'Add recovery target')
    saved=[]
    def capture(root,run,review,frames):
        dashboard.validate_review_for_run(review,frames);saved.append(review)
        return ROOT/'results'/'reviews'/'test_review.json'
    monkeypatch.setattr(dashboard,'save_review',capture)
    app.button(key='save_review').click().run()
    assert not app.exception
    assert saved[0]['events'][0]['frame_index']==20
    assert saved[0]['recovery_targets'][0]['player_label']=='Player A'
    assert saved[0]['source_video_sha256']


def test_apptest_analysis_requires_click_and_reports_process_failure(monkeypatch):
    calls=[]
    def failed(root,command,on_line):
        calls.append(command);on_line('Synthetic process failure for callback test');return 2
    monkeypatch.setattr(dashboard,'run_analysis',failed)
    app=make_app()
    assert calls==[]
    app.selectbox(key='analysis_video').set_value(ROOT/'data'/'lee_axelsen_rally.mp4').run()
    assert app.selectbox(key='analysis_video').value.name=='lee_axelsen_rally.mp4'
    assert not app.button(key='analyze_rally').disabled
    app.button(key='analyze_rally').click().run()
    assert not app.exception
    assert len(calls)==1 and '--milestone' in calls[0] and '--player-roi' in calls[0]
    assert calls[0][calls[0].index('--pose-nms-iou')+1]=='0.5'
    assert any('did not complete' in error.value for error in app.error)


def test_apptest_highlights_blocked_until_trimmed():
    app=make_app()
    source=ROOT/'data'/'lee_axelsen_50fps.mp4'
    if not source.is_file():pytest.skip('Full highlights source unavailable')
    app.selectbox(key='analysis_video').set_value(source).run()
    assert not app.exception
    assert app.button(key='analyze_rally').disabled
    assert any('long source' in warning.value.lower() for warning in app.warning)


def test_apptest_recalculate_updates_labels_without_inference(monkeypatch):
    app=choose_run(make_app(),'m3-sample')
    app.text_input(key='identity_label').input('Player A')
    app.selectbox(key='identity_side').select('near')
    press_form(app,'Add or update player label')
    calls=[]
    def recalculate(root,run,review,frames):
        from badminton_tracker.court import load_calibration
        from badminton_tracker.insights import analyze_insights
        calibration=load_calibration(run/'court_calibration.json',1280,720)
        insights=analyze_insights(dashboard.frames_of(dashboard.read_json(run/'shuttle.json')),frames,calibration,review=review)
        calls.append(review)
        return run/'review.json',insights
    monkeypatch.setattr(dashboard,'recompute_reviewed_insights',recalculate)
    app.button(key='recalculate_review').click().run()
    assert not app.exception
    assert len(calls)==1
    assert app.session_state['reviewed_insights_m3-sample']['identified_players'][0]['player_label']=='Player A'
    assert any(metric.label=='Distance retained' and metric.value=='4.11 m' for metric in app.metric)


def test_apptest_foreign_review_hash_is_rejected():
    app=choose_run(make_app(),'m3-sample')
    review={'schema_version':1,'source_video_sha256':'0'*64,'identity_map':{},'events':[],'rallies':[],'recovery_targets':[]}
    app.text_area(key='review_json_m3-sample').input(json.dumps(review))
    app.button(key='apply_review_json').click().run()
    assert not app.exception
    assert any('different source video' in error.value for error in app.error)
    assert app.session_state['review_m3-sample']['source_video_sha256']!='0'*64



def test_command_forwards_source_specific_overlap_suppression(tmp_path):
    (tmp_path/'data').mkdir();video=tmp_path/'data'/'clip.mp4';video.touch()
    python=tmp_path/'.venv'/'Scripts'/'python.exe';python.parent.mkdir(parents=True);python.touch()
    calibration=tmp_path/'configs'/'court.json';calibration.parent.mkdir();calibration.touch()
    command,_=dashboard.analysis_command(tmp_path,video,'singles',calibration,pose_nms_iou=.5)
    assert command[command.index('--pose-nms-iou')+1]=='0.5'
    doubles,_=dashboard.analysis_command(tmp_path,video,'doubles',calibration)
    assert doubles[doubles.index('--pose-nms-iou')+1]=='0.7'
    with pytest.raises(ValueError):dashboard.analysis_command(tmp_path,video,'singles',calibration,pose_nms_iou=float('nan'))
