{{/*
Expand the name of the chart.
*/}}
{{- define "helix.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "helix.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "helix.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "helix.labels" -}}
helm.sh/chart: {{ include "helix.chart" . }}
{{ include "helix.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: helix
{{- end }}

{{/*
Selector labels
*/}}
{{- define "helix.selectorLabels" -}}
app.kubernetes.io/name: {{ include "helix.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Aggregator labels
*/}}
{{- define "helix.aggregator.labels" -}}
{{ include "helix.labels" . }}
app.kubernetes.io/component: aggregator
{{- end }}

{{/*
Aggregator selector labels
*/}}
{{- define "helix.aggregator.selectorLabels" -}}
{{ include "helix.selectorLabels" . }}
app.kubernetes.io/component: aggregator
{{- end }}

{{/*
Worker labels
*/}}
{{- define "helix.worker.labels" -}}
{{ include "helix.labels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Worker selector labels
*/}}
{{- define "helix.worker.selectorLabels" -}}
{{ include "helix.selectorLabels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Dashboard labels
*/}}
{{- define "helix.dashboard.labels" -}}
{{ include "helix.labels" . }}
app.kubernetes.io/component: dashboard
{{- end }}

{{/*
Dashboard selector labels
*/}}
{{- define "helix.dashboard.selectorLabels" -}}
{{ include "helix.selectorLabels" . }}
app.kubernetes.io/component: dashboard
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "helix.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "helix.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the Ethereum RPC URL
*/}}
{{- define "helix.ethereumRpcUrl" -}}
{{- if .Values.ethereum.local.enabled }}
{{- printf "http://%s-ethereum:8545" (include "helix.fullname" .) }}
{{- else }}
{{- .Values.ethereum.external.rpcUrl }}
{{- end }}
{{- end }}

{{/*
Return the aggregator service URL
*/}}
{{- define "helix.aggregatorUrl" -}}
{{- printf "http://%s-aggregator:9001" (include "helix.fullname" .) }}
{{- end }}

{{/*
Create the aggregator P2P address
*/}}
{{- define "helix.aggregatorP2pAddr" -}}
{{- printf "%s-aggregator:9000" (include "helix.fullname" .) }}
{{- end }}

{{/*
Get image pull secrets
*/}}
{{- define "helix.imagePullSecrets" -}}
{{- with .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Common environment variables for all HELIX nodes
*/}}
{{- define "helix.commonEnv" -}}
- name: RUST_LOG
  value: "info,helix_node=debug"
- name: RUST_BACKTRACE
  value: "1"
- name: HELIX_ETH_RPC
  value: {{ include "helix.ethereumRpcUrl" . | quote }}
- name: HELIX_ETH_CHAIN_ID
  value: {{ .Values.ethereum.external.chainId | default 31337 | quote }}
{{- if .Values.contracts.coordinatorAddress }}
- name: HELIX_COORDINATOR_ADDRESS
  value: {{ .Values.contracts.coordinatorAddress | quote }}
{{- end }}
{{- end }}
