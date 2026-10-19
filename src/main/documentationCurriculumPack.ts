import { createHash } from "node:crypto";
import userGuide from "../../docs/v2/USER_GUIDE.md?raw";
import integrations from "../../docs/v2/INTEGRATIONS.md?raw";
import plugins from "../../docs/v2/plugins.md?raw";
import htmlArtifacts from "../../docs/v2/html-artifacts.md?raw";
import executableTrainingParsers from "../../docs/v2/executable-training-parsers.md?raw";
import managedEnvironments from "../../docs/v2/managed-environments.md?raw";
import clipboard from "../../docs/v2/clipboard.md?raw";
import apiTeacherModelDiscovery from "../../docs/v2/api-teacher-model-discovery.md?raw";

const PACK_ID = "omni-v2-user-documentation-23";
const PACK_VERSION = 23;
const LICENSE = "PolyForm-Noncommercial-1.0.0-or-commercial-license";
const REPOSITORY = "https://github.com/Eabusham2/Omni-AGI";

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
        .map(([key, child]) => [key, canonical(child)])
    );
  }
  return value;
}

function normalizedDocument(value: string): string {
  return `${value.replace(/\r\n?/g, "\n").trim()}\n`;
}

const documents = [
  {
    id: "user-guide",
    name: "Omni AGI Studio v2 user guide",
    path: "docs/v2/USER_GUIDE.md",
    content: normalizedDocument(userGuide)
  },
  {
    id: "integrations",
    name: "Omni AGI Studio v2 integrations reference",
    path: "docs/v2/INTEGRATIONS.md",
    content: normalizedDocument(integrations)
  },
  {
    id: "plugins",
    name: "Omni AGI Studio v2 plugin and custom-tool SDK",
    path: "docs/v2/plugins.md",
    content: normalizedDocument(plugins)
  },
  {
    id: "html-artifacts",
    name: "Omni AGI Studio v2 saved HTML artifact reference",
    path: "docs/v2/html-artifacts.md",
    content: normalizedDocument(htmlArtifacts)
  },
  {
    id: "executable-training-parsers",
    name: "Omni AGI Studio v2 executable training parser reference",
    path: "docs/v2/executable-training-parsers.md",
    content: normalizedDocument(executableTrainingParsers)
  },
  {
    id: "managed-environments",
    name: "Omni AGI Studio v2 managed sandbox and VM reference",
    path: "docs/v2/managed-environments.md",
    content: normalizedDocument(managedEnvironments)
  },
  {
    id: "clipboard",
    name: "Omni AGI Studio v2 instance and host clipboard reference",
    path: "docs/v2/clipboard.md",
    content: normalizedDocument(clipboard)
  },
  {
    id: "api-teacher-model-discovery",
    name: "Omni AGI Studio v2 API teacher model discovery reference",
    path: "docs/v2/api-teacher-model-discovery.md",
    content: normalizedDocument(apiTeacherModelDiscovery)
  }
] as const;

const declaration = {
  format: "omni-ordinary-document-curriculum",
  version: PACK_VERSION,
  id: PACK_ID,
  sourceKind: "project-authored-user-documentation",
  ordinaryNeuralLearning: true,
  hiddenPrompt: false,
  behavioralPersona: false,
  preferenceLabels: false,
  rewardModel: false,
  executedHostActions: false,
  license: LICENSE,
  repository: REPOSITORY,
  records: documents.map((document) => ({
    id: document.id,
    name: document.name,
    path: document.path,
    bytes: Buffer.byteLength(document.content),
    contentSha256: sha256(document.content),
    provenanceUrl: `${REPOSITORY}/blob/main/${document.path}`
  }))
} as const;

export type DocumentationCurriculumPack = ReturnType<
  typeof declaredDocumentationCurriculumPack
>;

export interface DocumentationCurriculumReceipt {
  format: "omni-documentation-curriculum-receipt";
  version: 1;
  packId: string;
  packVersion: number;
  packSha256: string;
  complete: true;
  records: Array<{
    id: string;
    contentSha256: string;
    trainingSourceId: string;
    duplicate: boolean;
    parametersChanged: boolean;
    learnedParameterSteps: number;
  }>;
}

/**
 * Versioned ordinary corpus material. This is intentionally separate from the
 * frozen origin-v3 action curriculum and contains no executable action target
 * or claim that a host tool succeeded.
 */
export function declaredDocumentationCurriculumPack() {
  const body = {
    ...declaration,
    records: declaration.records.map((record, index) => ({
      ...record,
      content: documents[index]!.content
    }))
  };
  return {
    ...body,
    sha256: sha256(JSON.stringify(canonical(declaration)))
  };
}
