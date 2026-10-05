export interface PlainRange {
  start: { line: number; character: number };
  end: { line: number; character: number };
}

export interface DocumentSnapshot {
  uri: string;
  version: number;
  range: PlainRange;
  selectionHash: string;
}

export interface CurrentDocumentState {
  uri: string;
  version: number;
  range: PlainRange;
  selectionHash: string;
}

export function sameRange(left: PlainRange, right: PlainRange): boolean {
  return (
    left.start.line === right.start.line &&
    left.start.character === right.start.character &&
    left.end.line === right.end.line &&
    left.end.character === right.end.character
  );
}

export function replacementTargetStillValid(snapshot: DocumentSnapshot | undefined, current: CurrentDocumentState): boolean {
  if (!snapshot) {
    return false;
  }
  return (
    snapshot.uri === current.uri &&
    snapshot.version === current.version &&
    snapshot.selectionHash === current.selectionHash &&
    sameRange(snapshot.range, current.range)
  );
}
