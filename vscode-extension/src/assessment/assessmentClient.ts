import { JsonRequester } from "../common/http";
import { AssessmentStatus, SessionStatus } from "../common/types";

export class AssessmentClient {
  constructor(private readonly requestJson: JsonRequester) {}

  status(): Promise<AssessmentStatus> {
    return this.requestJson<AssessmentStatus>("/api/v1/status");
  }

  session(): Promise<SessionStatus> {
    return this.requestJson<SessionStatus>("/api/v1/session");
  }
}
