import { describe, it } from "mocha"
import "should"
import { ZulongError, ZulongErrorType } from "../ZulongError"

describe("ZulongError", () => {
	describe("getErrorType", () => {
		it("should return QuotaExceeded when code is INFERENCE_CAP_ERROR", () => {
			const err = new ZulongError({ message: "Inference cap reached", code: "INFERENCE_CAP_ERROR" })
			ZulongError.getErrorType(err)!.should.equal(ZulongErrorType.QuotaExceeded)
		})
	})
})
