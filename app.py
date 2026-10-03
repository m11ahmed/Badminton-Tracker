"""Local Streamlit interface for Badminton Tracker."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import streamlit as st
from badminton_tracker.dashboard import (
    analysis_command, calibration_preview, contained_path, court_image, frames_of,
    inspect_video, list_runs, list_videos, load_run, position_rows, read_json,
    run_analysis, save_review, save_upload, source_frame, track_ids, extract_clip, recompute_reviewed_insights,
    write_manual_calibration, write_roi, prepared_pose_nms_iou,
)

st.set_page_config(page_title="Rally Lab | Badminton Tracker", page_icon="🏸", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1450px;}
.rally-hero {background:linear-gradient(115deg,#092c27,#123e35 65%,#1c5141);padding:28px 34px;border-radius:18px;color:#edf9f2;margin-bottom:22px;}
.rally-hero .eyebrow {font-size:12px;letter-spacing:3px;color:#77e6b4;font-weight:700;}
.rally-hero h1 {font-size:36px;line-height:1.15;margin:10px 0 9px;color:#f5fff8;}
.rally-hero p {margin:0;color:#c6ddce;font-size:16px;}
[data-testid="stMetric"] {border:1px solid rgba(90,140,117,.24);border-radius:12px;padding:14px;}
[data-testid="stMetricLabel"] {font-size:13px;}
[data-testid="stMetricValue"] {font-size:27px;}
[data-testid="stSidebar"] {border-right:1px solid rgba(90,140,117,.18);}
</style><div class="rally-hero"><div class="eyebrow">RALLY LAB / BADMINTON TRACKER</div><h1>See the rally.<br>Understand the movement.</h1><p>Shuttle tracking, court positioning and a place for your coaching notes.</p></div>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_video_info(path: str, modification_ns: int):
    return inspect_video(Path(path))


@st.cache_data(show_spinner=False, max_entries=40)
def cached_source_frame(path: str, modification_ns: int, frame_index: int):
    return source_frame(Path(path), frame_index)


def number(value, digits=2, unit=""):
    if value is None:
        return "Unavailable"
    try:
        return f"{float(value):.{digits}f}{unit}"
    except (ValueError, TypeError):
        return "Unavailable"


def show_review_editor(bundle, frames):
    run = bundle["path"]
    key = "review_" + run.name
    insights = bundle.get("insights") or {}
    if key not in st.session_state:
        initial = insights.get("review")
        if not isinstance(initial, dict) or "identity_map" not in initial:
            initial = {"schema_version":1, "identity_map":{}, "rallies":[], "events":[], "recovery_targets":[]}
        st.session_state[key] = copy.deepcopy(initial)
    review = st.session_state[key]
    source_hash = bundle["summary"].get("provenance",{}).get("input_video_sha256")
    if source_hash:
        if review.get("source_video_sha256") and review["source_video_sha256"] != source_hash:
            st.error("This review belongs to a different source video. Use the matching result before applying annotations.")
            return
        review.setdefault("source_video_sha256",source_hash)
    ids = track_ids(frames)
    if not ids:
        st.info("Player observations are needed before you can label athletes or add recovery targets.")
        return
    st.caption("Tracker IDs describe continuous fragments. Give fragments the same player name only after checking the video; statistics never bridge an ID gap automatically.")
    with st.form("identity_form_"+run.name):
        cols = st.columns([1,2,1,1])
        fragment = cols[0].selectbox("Tracker fragment", ids, format_func=lambda value:f"ID {value}", key="identity_fragment")
        label = cols[1].text_input("Player name", placeholder="e.g. Player A", key="identity_label")
        side = cols[2].selectbox("Court side", ["unassigned", "near", "far"], key="identity_side")
        team = cols[3].text_input("Team (optional)", key="identity_team")
        if st.form_submit_button("Add or update player label"):
            candidate = copy.deepcopy(review)
            if not label.strip():
                st.error("Enter a player name before saving a label.")
            else:
                mapping = {"player_label":label.strip()}
                if side != "unassigned": mapping["side"] = side
                if team.strip(): mapping["team"] = team.strip()
                candidate.setdefault("identity_map", {})[str(fragment)] = mapping
                try:
                    from badminton_tracker.insights import validate_review
                    validate_review(candidate, frames)
                    st.session_state[key] = candidate; review = candidate
                    st.success("Player label added to this review.")
                except (ImportError, ValueError) as exc:
                    st.error(str(exc))
    if review.get("identity_map"):
        st.dataframe([{"Tracker fragment":k, **v} for k,v in review["identity_map"].items()], hide_index=True, width="stretch")
    max_frame = max(int(frame["frame_index"]) for frame in frames)
    with st.form("contact_form_"+run.name):
        cols = st.columns([1,1,2])
        frame_index = cols[0].number_input("Contact frame", min_value=0, max_value=max_frame, value=0, key="contact_frame")
        kind = cols[1].selectbox("Annotation", ["contact", "outcome", "note"], key="contact_kind")
        player_label = cols[2].text_input("Player name for annotation", key="contact_player")
        note = st.text_input("Shot, outcome or coaching observation", placeholder="e.g. Late return from the rear court", key="contact_note")
        if st.form_submit_button("Add annotation"):
            event = {"frame_index":int(frame_index), "kind":kind, "note":note.strip()}
            if player_label.strip(): event["player_label"] = player_label.strip()
            candidate = copy.deepcopy(review)
            candidate.setdefault("events", []).append(event)
            try:
                from badminton_tracker.insights import validate_review
                validated = validate_review(candidate,frames)
                st.session_state[key] = validated; review = validated
                st.success("Annotation added. Save the review to retain it on disk.")
            except (ImportError,ValueError) as exc: st.error(str(exc))
    with st.expander("Set a recovery target or rally boundaries"):
        with st.form("target_form_"+run.name):
            target_label = st.text_input("Player name for target", key="target_player")
            cols = st.columns(4)
            contact = cols[0].number_input("After frame", 0, max_frame, 0, key="target_contact")
            until = cols[1].number_input("Review until frame", 0, max_frame, max_frame, key="target_until")
            x_m = cols[2].number_input("Target x (m)", 0.0, 6.1, 3.05, step=.05, key="target_x")
            y_m = cols[3].number_input("Target y (m)", 0.0, 13.4, 10.05, step=.05, key="target_y")
            radius = st.number_input("Acceptable radius (m)", .1, 3.0, .5, step=.1, key="target_radius")
            if st.form_submit_button("Add recovery target"):
                if not target_label.strip():
                    st.error("Enter the reviewed player name for this target.")
                elif until < contact:
                    st.error("The end frame must follow the contact frame.")
                else:
                    candidate = copy.deepcopy(review)
                    candidate.setdefault("recovery_targets", []).append({"player_label":target_label.strip(), "contact_frame":int(contact), "until_frame":int(until), "target_x_m":x_m, "target_y_m":y_m, "radius_m":radius})
                    try:
                        from badminton_tracker.insights import validate_review
                        validated = validate_review(candidate,frames)
                        st.session_state[key] = validated; review = validated
                        st.success("Recovery target added.")
                    except (ImportError,ValueError) as exc: st.error(str(exc))
        with st.form("rally_form_"+run.name):
            cols = st.columns(2)
            start = cols[0].number_input("Rally start frame", 0, max_frame, 0, key="rally_start")
            end = cols[1].number_input("Rally end frame", 0, max_frame, max_frame, key="rally_end")
            outcome = st.text_input("Rally outcome", placeholder="e.g. Player A winner", key="rally_outcome")
            if st.form_submit_button("Add rally boundary"):
                candidate = copy.deepcopy(review)
                candidate.setdefault("rallies", []).append({"id":f"rally{len(candidate.get('rallies',[]))+1}", "start_frame":int(start), "end_frame":int(end), "outcome":outcome.strip()})
                try:
                    from badminton_tracker.insights import validate_review
                    validate_review(candidate, frames)
                    st.session_state[key] = candidate; review = candidate
                    st.success("Rally boundary added.")
                except (ImportError, ValueError) as exc: st.error(str(exc))
    if review.get("events"):
        st.dataframe(review["events"], hide_index=True, width="stretch")
    with st.expander("Review annotations as JSON"):
        edited_review = st.text_area("Review JSON", json.dumps(review, indent=2), height=250, key="review_json_"+run.name)
        if st.button("Apply edited annotations", key="apply_review_json"):
            try:
                candidate = json.loads(edited_review)
                if source_hash and candidate.get("source_video_sha256") and candidate["source_video_sha256"] != source_hash:
                    raise ValueError("This review belongs to a different source video.")
                if source_hash: candidate["source_video_sha256"] = source_hash
                from badminton_tracker.insights import validate_review
                validated = validate_review(candidate, frames)
                st.session_state[key] = validated; review = validated
                st.success("Edited annotations applied to this review.")
            except (ValueError, ImportError) as exc: st.error(str(exc))
    review_text = json.dumps(review, indent=2)
    cols = st.columns(2)
    cols[0].download_button("Download review JSON", review_text, file_name=f"{run.name}_review.json", mime="application/json", key="download_review")
    if cols[1].button("Save coach review", key="save_review"):
        try:
            output = save_review(ROOT, run, review, frames)
            st.session_state["last_review_path"] = str(output)
            st.success(f"Saved a new review: {output.relative_to(ROOT)}")
        except (ImportError, ValueError, OSError) as exc:
            st.error(f"The review could not be saved: {exc}")
    if st.button("Recalculate insights with this review", key="recalculate_review"):
        try:
            with st.spinner("Recalculating from existing observations…"):
                output, reviewed_insights = recompute_reviewed_insights(ROOT, run, review, frames)
            st.session_state["reviewed_insights_"+run.name] = reviewed_insights
            st.session_state["reviewed_output_"+run.name] = str(output.parent)
            st.success("Reviewed insights saved separately. Open Movement and heatmaps to inspect the updated labels and targets.")
            st.rerun()
        except (ImportError, ValueError, OSError) as exc: st.error(str(exc))
    st.caption("Saving and recalculating create new files under results/reviews. Original tracking and measurements remain available.")


def review_tab(run):
    try:
        bundle = load_run(run)
    except (OSError, ValueError, TypeError) as exc:
        st.error(f"This result could not be opened: {exc}")
        return
    summary = bundle["summary"]
    provenance = summary.get("provenance", {})
    video_info = provenance.get("video", {})
    frames = frames_of(bundle.get("players"))
    shuttle = frames_of(bundle.get("shuttle"))
    ids = track_ids(frames)
    insights = st.session_state.get("reviewed_insights_"+run.name) or bundle.get("insights") or {}
    insight_folder = Path(st.session_state.get("reviewed_output_"+run.name, str(run)))
    cols = st.columns(4)
    cols[0].metric("Recorded frame rate", number(video_info.get("fps"), 0, " fps"))
    observed_times = [float(item.get("timestamp_s",0)) for item in (shuttle or frames)]
    analyzed_span = (max(observed_times)-min(observed_times)+1/float(video_info.get("fps") or 1)) if observed_times else None
    cols[1].metric("Analyzed duration", number(analyzed_span, 2, " s"))
    cols[2].metric("Frames reviewed", len(shuttle) or len(frames))
    cols[3].metric("Tracker fragments", len(ids))
    st.caption(f"Result: {run.name} · {provenance.get('milestone','analysis')} · {provenance.get('settings',{}).get('players_mode','singles').capitalize()}")
    if (run / "annotated.mp4").is_file():
        st.video(str(run / "annotated.mp4"))
        st.caption("The annotated video includes the court map inset. Use Positioning below to scrub the original frame and matching court diagram together.")
    elif (run/"camera_edge_departure_review.jpg").is_file():
        st.image(str(run/"camera_edge_departure_review.jpg"),caption="Player validation: source frames and predicted poses",width="stretch")
        st.caption("This validation run analyzed players only. Shuttle tracking and court calibration were not run.")
    else:
        st.info("An annotated video is unavailable for this result.")
    if (run / "minimap.mp4").is_file():
        st.download_button(
            "Download standalone court map video",
            data=(run / "minimap.mp4").read_bytes(),
            file_name=f"{run.name}_court_map.mp4",
            mime="video/mp4",
            key=f"download_minimap_{run.name}",
        )
    warnings = insights.get("quality", {}).get("warnings", []) if isinstance(insights.get("quality"), dict) else []
    if warnings:
        for warning in warnings:
            st.warning(str(warning))
    if frames and any(frame.get("count_mismatch") for frame in frames):
        st.warning("Some frames contain fewer or more players than expected. Check the video before interpreting individual totals.")
    if (bundle.get("court_calibration") or {}).get("method", "").startswith("auto") and not (bundle.get("court_calibration") or {}).get("metadata",{}).get("manually_reviewed"):
        st.warning("The automatically proposed court corners need checking against the painted lines.")
    positions, movement, shuttle_tab, notes = st.tabs(["Positioning", "Movement & heatmaps", "Shuttle motion", "Coach review"])
    with positions:
        rows = position_rows(frames)
        if not rows:
            st.info("Court positions are available after M3 court calibration. You can still review the tracking video.")
        else:
            max_frame = max(int(frame["frame_index"]) for frame in frames)
            frame_index = st.slider("Review frame", 0, max_frame, 0, key="review_frame") if max_frame > 0 else 0
            frame = next((item for item in frames if item["frame_index"] == frame_index), None)
            timestamp = frame.get("timestamp_s", 0) if frame else 0
            st.caption(f"Source frame {frame_index} · {timestamp:.3f} s. Both images below refer to this frame.")
            image_cols = st.columns([4,1])
            source_path = provenance.get("input_video")
            try:
                source = contained_path(ROOT, source_path, "data") if source_path else None
                if source and source.is_file():
                    image_cols[0].image(cached_source_frame(str(source), source.stat().st_mtime_ns, frame_index), width="stretch")
                else:
                    image_cols[0].info("The original source video is unavailable in the data folder.")
            except (ValueError, OSError) as exc:
                image_cols[0].info(str(exc))
            image_cols[1].image(court_image(rows, frame_index=frame_index), width="stretch")
            st.dataframe([row for row in rows if row["frame_index"] == frame_index], hide_index=True, width="stretch")
            st.caption("Positions approximate the players' ankle midpoints on the court floor. A jump, occlusion or inaccurate corner selection can change the estimate.")
    with movement:
        players = insights.get("players", [])
        if not players:
            st.info("This result contains tracking and positioning only. Analyze with M5 to calculate movement and heatmaps.")
        else:
            table = [{"Fragment":f"ID {player['track_id']}", "Reviewed name":player.get("player_label") or "Unassigned", "Distance (m)":player.get("distance_m"), "Average speed (m/s)":player.get("average_speed_m_s"), "Max retained speed (m/s)":player.get("max_speed_m_s"), "Measured movement (s)":player.get("valid_motion_time_s")} for player in players]
            st.dataframe(table, hide_index=True, width="stretch")
            identified = insights.get("identified_players",[])
            if identified:
                st.markdown("**Reviewed player totals**")
                st.dataframe([{"Player":item["player_label"],"Tracker fragments":", ".join(str(value) for value in item.get("track_ids",[])),"Distance (m)":item.get("distance_m"),"Average speed (m/s)":item.get("average_speed_m_s"),"Measured movement (s)":item.get("valid_motion_time_s")} for item in identified],hide_index=True,width="stretch")
            chosen = st.selectbox("Inspect tracker fragment", [int(p["track_id"]) for p in players], format_func=lambda value:f"ID {value}", key="movement_fragment")
            player = next(p for p in players if int(p["track_id"]) == chosen)
            cols = st.columns(3)
            cols[0].metric("Distance retained", number(player.get("distance_m"), 2, " m"))
            cols[1].metric("Average movement speed", number(player.get("average_speed_m_s"), 2, " m/s"))
            cols[2].metric("Measured movement", number(player.get("valid_motion_time_s"), 2, " s"))
            heatmap = insight_folder / "heatmaps" / f"track_{chosen}.png"
            heat_cols = st.columns([1,2])
            if heatmap.is_file(): heat_cols[0].image(str(heatmap), caption="Time spent by court region", width="stretch")
            else: heat_cols[0].image(court_image(position_rows(frames), selected_id=chosen), caption="Recorded position trail", width="stretch")
            zones = player.get("zones", {})
            if zones:
                zone_rows = [{"Court zone":name, "Time (s)":value.get("seconds", value.get("duration_s",0)) if isinstance(value,dict) else value} for name,value in zones.items()]
                heat_cols[1].dataframe(zone_rows, hide_index=True, width="stretch")
            motion = [row for row in insights.get("player_motion", []) if int(row.get("track_id", -1)) == chosen]
            speed_key = next((key for key in ("speed_m_s", "raw_speed_m_s") if any(key in row for row in motion)), None)
            if speed_key and motion:
                st.line_chart([{ "time_s":row.get("timestamp_s",0), "speed_m_s":row.get(speed_key)} for row in motion], x="time_s", y="speed_m_s", x_label="Time (s)", y_label="Speed (m/s)")
            st.caption("Distance and speed use retained continuous observations. Gaps and implausible jumps are excluded. Totals describe this clip and are not a fitness or skill score.")
            spacing = insights.get("doubles_pair_spacing", {})
            if spacing:
                if spacing.get("status") == "available":
                    st.subheader("Doubles partnership spacing")
                    st.dataframe([{"Court side":pair.get("side"),"Reviewed partners":" + ".join(pair.get("players",[])),"Valid frames":pair.get("valid_frames"),"Median spacing (m)":pair.get("median_spacing_m"),"Minimum spacing (m)":pair.get("minimum_spacing_m"),"Maximum spacing (m)":pair.get("maximum_spacing_m")} for pair in spacing.get("pairs",[])], hide_index=True, width="stretch")
                elif provenance.get("settings",{}).get("players_mode") == "doubles":
                    st.info(spacing.get("reason") or "Assign reviewed near and far court player labels to calculate partnership spacing.")
    with shuttle_tab:
        st.info("Shuttle motion is measured in image pixels per second. Actual speed in km/h requires additional calibrated 3D information: the shuttle flies above the court floor.")
        shuttle_summary = insights.get("shuttle", {})
        cols = st.columns(3)
        cols[0].metric("Average image speed", number(shuttle_summary.get("average_image_speed_px_s"), 0, " px/s"))
        cols[1].metric("Largest detected image motion", number(shuttle_summary.get("max_image_speed_px_s"), 0, " px/s"))
        cols[2].metric("Physical shuttle speed", "Unavailable")
        motion = insights.get("shuttle_motion", [])
        if motion:
            st.line_chart([{ "time_s":row.get("timestamp_s",0), "image_speed_px_s":row.get("image_speed_px_s")} for row in motion], x="time_s", y="image_speed_px_s", x_label="Time (s)", y_label="Image speed (px/s)")
        elif not insights:
            st.caption("M5 analysis adds the image speed timeline.")
        counts = {name:sum(row.get("source") == name for row in shuttle) for name in ("detected", "interpolated", "missing")}
        st.dataframe([{"Shuttle observation":name.capitalize(), "Frames":count} for name,count in counts.items()], hide_index=True, width="stretch")
        st.caption("Abrupt jumps are flagged for review, not erased. A large image speed can be a false detection or perspective effect; it is not a fastest shot estimate. Compare footage only within this camera view.")
    with notes:
        if insights.get("rallies"):
            boundaries_reviewed = bool(insights.get("review",{}).get("rallies"))
            if not boundaries_reviewed:
                st.info("No rally boundaries reviewed. These totals cover the selected clip span.")
            st.markdown("**Reviewed rallies**" if boundaries_reviewed else "**Clip span**")
            rally_rows = [{"Rally":rally.get("id","Reviewed rally") if boundaries_reviewed else "Clip span", "Start frame":rally.get("start_frame"), "End frame":rally.get("end_frame"), "Duration (s)":rally.get("duration_s"), "Outcome":rally.get("outcome") or ("Winner: "+rally["winner_label"] if rally.get("winner_label") else "Not reviewed"), "Notes":rally.get("notes") or ""} for rally in insights["rallies"]]
            st.dataframe(rally_rows,hide_index=True,width="stretch")
        if insights.get("recovery"):
            st.markdown("**Recovery to your targets**")
            st.dataframe(insights["recovery"],hide_index=True,width="stretch")
            st.caption("Recovery timing is reported only while a reviewed player has a continuous valid trajectory. A gap or fragment change makes the timing unavailable.")
        reviewed_events = insights.get("reviewed_events",[])
        if reviewed_events:
            event_rows=[]
            for event in reviewed_events:
                positions=event.get("positions",[])
                event_rows.append({"Frame":event["frame_index"],"Annotation":event.get("kind"),"Player":event.get("player_label"),"Note":event.get("note"),"Court positions":", ".join(f"ID {position['track_id']}: ({number(position.get('x_m'))}, {number(position.get('y_m'))}) m" for position in positions)})
            st.dataframe(event_rows,hide_index=True,width="stretch")
        show_review_editor(bundle, frames)
    with st.expander("Export measurements"):
        for name in ("insights.json", "court_positions.csv", "player_motion.csv", "shuttle_motion.csv", "players.csv", "shuttle.csv", "coaching_review.md"):
            path = (insight_folder if name in {"insights.json","player_motion.csv","shuttle_motion.csv","coaching_review.md"} else run) / name
            if path.is_file():
                st.download_button(f"Download {name}", path.read_bytes(), file_name=f"{run.name}_{name}", key="export_"+name)


def analyze_tab():
    st.subheader("Analyze a fixed camera rally")
    st.write("Choose an uninterrupted rally with the complete court in view. For a highlights video, prepare a rally clip first; camera cuts and replays need separate calibration.")
    videos = list_videos()
    if st.session_state.get("newly_saved_video"):
        new_video = Path(st.session_state.pop("newly_saved_video"))
        if new_video in videos: st.session_state["analysis_video"] = new_video
    selected = st.selectbox("Video in the project data folder", videos, format_func=lambda path:str(path.relative_to(ROOT / "data")), key="analysis_video") if videos else None
    with st.expander("Upload another local video"):
        upload = st.file_uploader("Video file", type=["mp4", "mov", "mkv", "avi"], key="source_upload")
        if st.button("Save upload to project", disabled=upload is None, key="save_upload"):
            try:
                saved = save_upload(ROOT, upload.name, upload.getvalue())
                st.session_state["newly_saved_video"] = str(saved)
                st.rerun()
            except (ValueError, OSError) as exc: st.error(str(exc))
    if selected is None:
        st.info("Place a video in the project's data folder or upload one above.")
        return
    try:
        metadata = cached_video_info(str(selected), selected.stat().st_mtime_ns)
    except (ValueError, OSError) as exc:
        st.error(f"The video could not be read: {exc}")
        return
    st.caption(f"{metadata['width']} × {metadata['height']} · {metadata['fps']:.2f} fps · {number(metadata.get('duration_s'),2,' s')}")
    first_frame = cached_source_frame(str(selected), selected.stat().st_mtime_ns, 0)
    st.image(first_frame, caption="First frame of the chosen source", width="stretch")
    with st.expander("Extract one rally from this video", expanded=bool((metadata.get("duration_s") or 0)>45)):
        st.caption("Enter seconds in the source video. Choose a fixed court view without a replay or camera cut. Extraction preserves the source frame timing.")
        with st.form("trim_form"):
            trim_cols = st.columns(2)
            trim_start = trim_cols[0].number_input("Rally start (s)", min_value=0.0, value=0.0, step=.02, key="trim_start")
            trim_end = trim_cols[1].number_input("Rally end (s)", min_value=.02, value=min(float(metadata.get("duration_s") or 10),10.), step=.02, key="trim_end")
            if st.form_submit_button("Extract rally clip"):
                try:
                    with st.status("Extracting the selected rally…", expanded=True) as status:
                        log_area = st.empty(); lines=[]
                        def update(line):
                            lines.append(line); log_area.code("\n".join(lines[-12:]), language="text")
                        extracted = extract_clip(ROOT,selected,float(trim_start),float(trim_end),update)
                        status.update(label="Rally clip ready",state="complete")
                    st.session_state["newly_saved_video"] = str(extracted)
                    st.rerun()
                except (ValueError,OSError) as exc: st.error(str(exc))
    long_source = (metadata.get("duration_s") or 0)>45
    if long_source: st.warning("This is a long source. Extract an uninterrupted rally above before running neural analysis.")
    mode = st.radio("Match format", ["singles", "doubles"], format_func=lambda value:"Singles · 2 players" if value == "singles" else "Doubles · 4 players", horizontal=True, key="analysis_mode")
    calibration_method = st.radio("Court calibration", ["Existing calibration", "Enter four corners", "Automatic proposal"], horizontal=True, key="calibration_method")
    calibration = None
    corners = None
    if calibration_method == "Existing calibration":
        configs = []
        for path in sorted((ROOT / "configs").glob("*.json")):
            try:
                document = read_json(path)
                if isinstance(document,dict) and document.get("source_dimensions") == [metadata['width'],metadata['height']] and "corners_px" in document:
                    bound_source = document.get("metadata",{}).get("input_video")
                    if not bound_source or Path(bound_source).resolve() == selected.resolve(): configs.append(path)
            except (OSError, ValueError): pass
        if configs:
            calibration = st.selectbox("Matching court calibration", configs, format_func=lambda p:p.name, key="calibration_config")
        else:
            st.info("No matching calibration is saved for this source. Enter its four outer court corners or create an automatic proposal.")
    elif calibration_method == "Enter four corners":
        st.caption("Use the outer doubles lines for both match formats. Coordinates refer to original source pixels. Order: far left, far right, near right, near left.")
        width, height = metadata['width'],metadata['height']
        defaults = [[.35*width,.36*height],[.65*width,.36*height],[.78*width,.90*height],[.22*width,.90*height]]
        corners = []
        for index, (label, values) in enumerate(zip(("Far left", "Far right", "Near right", "Near left"), defaults)):
            cols = st.columns([1,2,2]); cols[0].write(label)
            x = cols[1].number_input(f"{label} x", 0.0, float(width-1), float(values[0]), step=1., key=f"corner_x_{selected.name}_{index}")
            y = cols[2].number_input(f"{label} y", 0.0, float(height-1), float(values[1]), step=1., key=f"corner_y_{selected.name}_{index}")
            corners.append([x,y])
        st.image(calibration_preview(first_frame, corners), caption="Adjust the yellow corners onto the painted outer intersections", width="stretch")
    else:
        st.warning("Automatic corner detection is a proposal. Review its alignment after analysis before trusting court distances.")
    roi_options = [None]
    for path in sorted((ROOT / "configs").glob("*roi*.json")):
        roi_options.append(path)
    matching_roi = next((path for path in roi_options if path and path.name in {selected.stem+"_roi.json",selected.stem+"_player_roi.json"}),None)
    roi = st.selectbox("Player selection region", roi_options, index=roi_options.index(matching_roi) if matching_roi else 0, format_func=lambda p:p.name if p else "No region: tracking only, movement excluded", key="analysis_roi_"+selected.stem)
    if roi is None:
        st.warning("Movement summaries require a player selection region. Without one, people outside the court may be tracked and movement totals are withheld.")
    else:
        roi_data = read_json(roi)
        roi_corners = [[point[0]*metadata["width"],point[1]*metadata["height"]] for point in roi_data.get("points",[])]
        if len(roi_corners)==4:
            with st.expander("Check player selection region"):
                st.image(calibration_preview(first_frame,roi_corners),caption="The polygon should include active players and their feet, and exclude officials and spectators.",width="stretch")
    manual_roi_points = None
    if st.checkbox("Enter a player region for this source",key="manual_roi_"+selected.stem):
        st.caption("Normalized coordinates run from 0 to 1 across the source image. Include the floor around the active court, with enough apron for footwork; exclude people beyond it.")
        manual_roi_points = []
        defaults = [[.25,.30],[.75,.30],[.90,1.],[.10,1.]]
        for index,(label,default) in enumerate(zip(("Far left","Far right","Near right","Near left"),defaults)):
            cols=st.columns(2)
            rx=cols[0].number_input(f"Region {label} x",0.,1.,float(default[0]),step=.01,key=f"roi_x_{selected.stem}_{index}")
            ry=cols[1].number_input(f"Region {label} y",0.,1.,float(default[1]),step=.01,key=f"roi_y_{selected.stem}_{index}")
            manual_roi_points.append([rx,ry])
        st.image(calibration_preview(first_frame,[[x*metadata["width"],y*metadata["height"]] for x,y in manual_roi_points]),caption="Proposed player selection region",width="stretch")
    frame_limit = st.number_input("Frame limit (0 analyzes the whole clip)", min_value=0, max_value=100000, value=0, step=50, key="analysis_frame_limit")
    ready = (calibration_method != "Existing calibration" or calibration is not None) and not long_source
    st.caption("This computer runs neural tracking on CPU. A short rally can take several minutes. The log below reports progress; keep this browser tab and terminal open.")
    if st.button("Analyze rally", type="primary", disabled=not ready, key="analyze_rally"):
        try:
            if corners is not None:
                calibration = write_manual_calibration(ROOT, selected, corners)
            if manual_roi_points is not None:
                roi = write_roi(ROOT,manual_roi_points)
            command, output = analysis_command(ROOT, selected, mode, calibration, automatic=calibration_method == "Automatic proposal", max_frames=int(frame_limit), roi=roi, pose_nms_iou=prepared_pose_nms_iou(selected, mode))
            with st.status("Tracking the rally…", expanded=True) as status:
                log_area = st.empty(); lines = []
                def update(line):
                    lines.append(line)
                    log_area.code("\n".join(lines[-22:]), language="text")
                result = run_analysis(ROOT, command, update)
                if result != 0:
                    status.update(label="Analysis stopped with an error", state="error")
                    st.error("The analysis did not complete. The log above explains the issue; previous result folders remain available.")
                elif not (output / "summary.json").is_file() or read_json(output / "summary.json").get("status") != "complete":
                    status.update(label="Analysis output could not be verified", state="error")
                    st.error("The process finished without a complete result summary. Check the log before using this run.")
                else:
                    status.update(label="Rally analysis complete", state="complete", expanded=False)
                    st.success(f"Completed {output.name}. Refresh results, then open it in Rally review.")
                    st.session_state["completed_demo_run"] = str(output)
        except (ValueError, OSError) as exc:
            st.error(str(exc))


runs = list_runs()
with st.sidebar:
    st.markdown("### Your rally workspace")
    selected_run = st.selectbox("Completed analysis", runs, format_func=lambda path:path.name, key="selected_run") if runs else None
    if st.button("Refresh results", key="refresh_results"):
        st.rerun()
    st.caption("Local files stay in this project's G drive folder.")
    st.divider()
    st.markdown("**Singles and doubles**")
    st.caption("Choose two or four expected players. Review labels before combining fragments into player totals.")
    st.markdown("**Racket analysis**")
    st.caption("Deferred from this release.")
review_page, analysis_page, guide = st.tabs(["Rally review", "Analyze a clip", "Reading the insights"])
with review_page:
    if selected_run is not None:
        review_tab(selected_run)
    else:
        st.info("No completed analyses yet. Choose a video in Analyze a clip to create your first result.")
with analysis_page:
    analyze_tab()
with guide:
    st.subheader("Start with the video, then the numbers")
    st.write("Compare positioning and recovery with the rally context. More distance or a higher average speed does not by itself mean better badminton.")
    st.markdown("1. Check that shuttle markers and player poses follow the correct objects.\n2. Check court corners against the floor lines.\n3. Label each visible tracker fragment with the correct player.\n4. Review court zones, movement gaps and rally outcomes together.\n5. Add contacts and recovery targets to make your coaching question concrete.")
    st.write("Singles uses the inner sidelines and doubles uses the outer sidelines. Doubles partnership spacing requires reviewed player names and court sides, plus all four visible positions.")
    st.write("Shuttle speed in this release means image motion in pixels per second. A floor homography cannot recover the true speed of an airborne shuttle from one camera. Physical km/h stays unavailable.")
