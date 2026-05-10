@router.get("/", response_model=list[SessionListResponse])
def list_sessions(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    rows = (
        db.query(SessionModel, Report)
        .outerjoin(Report, Report.session_id == SessionModel.id)
        .filter(SessionModel.user_id == current_user.id)
        .order_by(SessionModel.created_at.desc())
        .all()
    )
    result = []
    for session, report in rows:
        shot_pct = None
        if report and report.total_shots and report.total_shots > 0:
            shot_pct = round(report.makes / report.total_shots * 100, 1)
        result.append(SessionListResponse(
            id=session.id,
            original_filename=session.original_filename,
            status=session.status,
            created_at=session.created_at,
            shot_percentage=shot_pct,
        ))
    return result
