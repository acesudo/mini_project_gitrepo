// Face Attendance Dashboard Client Logic
const API_BASE = window.location.origin.includes("localhost") || window.location.origin.includes("127.0.0.1")
  ? "" 
  : "http://localhost:8000";

let cameraStream = null;
let autoScanTimer = null;
let capturedSamples = []; // Array of Blobs for registration

// Elements
const cameraVideo = document.getElementById("cameraVideo");
const overlayCanvas = document.getElementById("overlayCanvas");
const enrollVideo = document.getElementById("enrollVideo");

// Initialize on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  checkServerHealth();
  initYearMonthSelectors();
  loadTodayAttendance();
  loadStudentDirectory();
  setupEventListeners();

  // Polling server health every 15s
  setInterval(checkServerHealth, 15000);
});

// ==================== TABS NAVIGATION ====================
function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      const tabId = btn.getAttribute("data-tab");
      switchTab(tabId);
    });
  });
}

function switchTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.getAttribute("data-tab") === tabId));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.id === `tab-${tabId}`));

  if (tabId === "kiosk") {
    startCamera();
  } else if (tabId === "enroll") {
    startEnrollCamera();
  } else if (tabId === "today") {
    loadTodayAttendance();
  } else if (tabId === "directory") {
    loadStudentDirectory();
  } else if (tabId === "monthly") {
    fetchMonthlySummary();
  }
}

// ==================== SERVER HEALTH ====================
async function checkServerHealth() {
  const statusEl = document.getElementById("serverStatus");
  const textEl = document.getElementById("serverStatusText");
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      statusEl.className = "server-status online";
      textEl.textContent = "API Online";
    } else {
      throw new Error();
    }
  } catch {
    statusEl.className = "server-status offline";
    textEl.textContent = "API Offline";
  }
}

// ==================== WEBRTC CAMERA ====================
async function startCamera() {
  if (cameraStream) return;
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } }
    });
    cameraVideo.srcObject = cameraStream;
    document.getElementById("cameraPlaceholder").style.display = "none";
    document.getElementById("btnToggleCamera").textContent = "Stop Camera";
    document.getElementById("btnScanOnce").disabled = false;

    if (document.getElementById("chkAutoScan").checked) {
      startAutoScan();
    }
  } catch (err) {
    showToast("Camera access denied or unavailable: " + err.message, "danger");
  }
}

function stopCamera() {
  stopAutoScan();
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
    cameraVideo.srcObject = null;
  }
  document.getElementById("cameraPlaceholder").style.display = "flex";
  document.getElementById("btnToggleCamera").textContent = "Start Camera";
  document.getElementById("btnScanOnce").disabled = true;

  const ctx = overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function startAutoScan() {
  stopAutoScan();
  autoScanTimer = setInterval(() => {
    if (cameraStream && !document.hidden) {
      scanFrame();
    }
  }, 2000);
}

function stopAutoScan() {
  if (autoScanTimer) {
    clearInterval(autoScanTimer);
    autoScanTimer = null;
  }
}

// ==================== FRAME RECOGNITION ====================
async function scanFrame() {
  if (!cameraStream || cameraVideo.videoWidth === 0) return;

  const indicator = document.getElementById("scanIndicator");
  indicator.style.display = "inline-block";

  // Capture frame to canvas
  const canvas = document.createElement("canvas");
  canvas.width = cameraVideo.videoWidth;
  canvas.height = cameraVideo.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(cameraVideo, 0, 0);

  canvas.toBlob(async blob => {
    indicator.style.display = "none";
    if (!blob) return;

    const formData = new FormData();
    formData.append("image", blob, "frame.jpg");
    formData.append("mark_attendance", "true");

    try {
      const res = await fetch(`${API_BASE}/recognition/frame`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Recognition endpoint error");
      const data = await res.json();

      drawBoundingBoxes(data.results, canvas.width, canvas.height);
      handleRecognitionResults(data.results);
    } catch (err) {
      console.warn("Scan failed:", err);
    }
  }, "image/jpeg", 0.85);
}

function drawBoundingBoxes(results, videoW, videoH) {
  overlayCanvas.width = cameraVideo.clientWidth;
  overlayCanvas.height = cameraVideo.clientHeight;
  const ctx = overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  const scaleX = overlayCanvas.width / videoW;
  const scaleY = overlayCanvas.height / videoH;

  results.forEach(item => {
    if (!item.bbox) return;
    const [x, y, w, h] = item.bbox;
    const rx = x * scaleX;
    const ry = y * scaleY;
    const rw = w * scaleX;
    const rh = h * scaleY;

    ctx.lineWidth = 3;
    ctx.strokeStyle = item.is_known ? "#10b981" : "#ef4444";
    ctx.strokeRect(rx, ry, rw, rh);

    const label = item.is_known ? `${item.student_id} (${(item.confidence * 100).toFixed(0)}%)` : "Unknown";
    ctx.font = "bold 13px sans-serif";
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = item.is_known ? "#10b981" : "#ef4444";
    ctx.fillRect(rx, Math.max(0, ry - 22), textWidth + 8, 22);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, rx + 4, Math.max(15, ry - 6));
  });
}

let lastAnnounced = {}; // Prevent spamming notifications every 2s
function handleRecognitionResults(results) {
  if (results.length === 0) return;

  const logContainer = document.getElementById("liveDetectionLog");
  const now = Date.now();

  results.forEach(r => {
    const key = r.student_id;
    if (lastAnnounced[key] && now - lastAnnounced[key] < 10000) return;
    lastAnnounced[key] = now;

    // Toast
    if (r.is_known) {
      if (r.attendance && r.attendance.marked) {
        showToast(`✅ Attendance Marked: ${r.student_id}`, "success");
        loadTodayAttendance();
      } else if (r.attendance && r.attendance.already_marked) {
        showToast(`ℹ️ Already Marked Today: ${r.student_id}`, "warning");
      }
    } else {
      showToast("⚠️ Unregistered / Unknown Face Detected", "danger");
    }

    // Add to Log card
    const card = document.createElement("div");
    card.style.padding = "0.75rem";
    card.style.background = r.is_known ? "var(--success-bg)" : "var(--danger-bg)";
    card.style.borderLeft = `4px solid ${r.is_known ? "var(--success)" : "var(--danger)"}`;
    card.style.borderRadius = "6px";
    card.style.fontSize = "0.85rem";
    card.innerHTML = `
      <div style="font-weight: 700;">${r.is_known ? r.student_id : "UNKNOWN PERSON"}</div>
      <div style="color: var(--text-muted);">Confidence: ${(r.confidence * 100).toFixed(1)}% | ${new Date().toLocaleTimeString()}</div>
      <div style="font-size: 0.8rem; margin-top: 2px;">${r.attendance ? r.attendance.message : "Not logged"}</div>
    `;

    if (logContainer.children[0] && logContainer.children[0].tagName === "P") {
      logContainer.innerHTML = "";
    }
    logContainer.prepend(card);
    if (logContainer.children.length > 20) logContainer.removeChild(logContainer.lastChild);
  });
}

// ==================== TODAY'S ATTENDANCE TAB ====================
let todayRecords = [];
async function loadTodayAttendance() {
  const dateInput = document.getElementById("filterDate");
  const onDate = dateInput.value || new Date().toISOString().split("T")[0];
  dateInput.value = onDate;

  try {
    const res = await fetch(`${API_BASE}/attendance/daily?date=${onDate}`);
    todayRecords = await res.json();
    renderTodayTable();
    updateKPICards();
    populateDepartmentFilters();
  } catch (err) {
    console.error("Failed to load today's attendance:", err);
  }
}

function renderTodayTable() {
  const tbody = document.getElementById("todayTableBody");
  const deptFilter = document.getElementById("filterDeptToday").value;
  const search = document.getElementById("searchToday").value.toLowerCase();

  const filtered = todayRecords.filter(r => {
    const matchDept = !deptFilter || r.department === deptFilter;
    const matchSearch = !search || 
      (r.student_id && r.student_id.toLowerCase().includes(search)) ||
      (r.student_name && r.student_name.toLowerCase().includes(search));
    return matchDept && matchSearch;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No attendance records found for this date.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(r => `
    <tr>
      <td><strong>${r.student_id}</strong></td>
      <td>${r.student_name || "—"}</td>
      <td>${r.department || "—"}</td>
      <td>${r.semester || "—"}</td>
      <td>${r.time || "—"}</td>
      <td><span class="badge badge-success">${r.status}</span></td>
      <td>${r.confidence ? (r.confidence * 100).toFixed(1) + "%" : "Manual"}</td>
    </tr>
  `).join("");
}

async function updateKPICards() {
  try {
    const stuRes = await fetch(`${API_BASE}/students`);
    const allStudents = await stuRes.json();
    const total = allStudents.length;
    const present = todayRecords.length;
    const absent = Math.max(0, total - present);
    const rate = total > 0 ? ((present / total) * 100).toFixed(0) + "%" : "0%";

    document.getElementById("kpiTotalStudents").textContent = total;
    document.getElementById("kpiPresentToday").textContent = present;
    document.getElementById("kpiAbsentToday").textContent = absent;
    document.getElementById("kpiRate").textContent = rate;
  } catch (err) {
    console.warn("KPI update error:", err);
  }
}

// ==================== STUDENT DIRECTORY ====================
let allStudentsList = [];
async function loadStudentDirectory() {
  try {
    const res = await fetch(`${API_BASE}/students`);
    allStudentsList = await res.json();
    renderDirectoryTable();
    populateDepartmentFilters();
  } catch (err) {
    console.error("Failed to load students:", err);
  }
}

function renderDirectoryTable() {
  const tbody = document.getElementById("directoryTableBody");
  const deptFilter = document.getElementById("filterDeptDirectory").value;
  const search = document.getElementById("searchDirectory").value.toLowerCase();

  const filtered = allStudentsList.filter(s => {
    const matchDept = !deptFilter || s.department === deptFilter;
    const matchSearch = !search ||
      s.student_id.toLowerCase().includes(search) ||
      s.name.toLowerCase().includes(search);
    return matchDept && matchSearch;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No students registered yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(s => `
    <tr>
      <td><strong>${s.student_id}</strong></td>
      <td>${s.name}</td>
      <td>${s.department}</td>
      <td>Semester ${s.semester}</td>
      <td>
        <span class="badge ${s.has_embedding ? 'badge-success' : 'badge-warning'}">
          ${s.has_embedding ? 'Biometrics Enrolled' : 'No Face Data'}
        </span>
      </td>
      <td>${s.created_at ? s.created_at.split("T")[0] : "—"}</td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick="viewHistory('${s.student_id}')">History</button>
        <button class="btn btn-danger btn-sm" onclick="deleteStudent('${s.student_id}')">Delete</button>
      </td>
    </tr>
  `).join("");
}

async function viewHistory(studentId) {
  document.getElementById("modalStudentTitle").textContent = `Attendance History: ${studentId}`;
  const tbody = document.getElementById("modalHistoryTableBody");
  tbody.innerHTML = `<tr><td colspan="4" style="text-align: center;">Loading history...</td></tr>`;
  document.getElementById("historyModal").classList.add("active");

  try {
    const res = await fetch(`${API_BASE}/attendance/student/${studentId}`);
    const history = await res.json();
    if (history.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No past attendance records found.</td></tr>`;
      return;
    }
    tbody.innerHTML = history.map(h => `
      <tr>
        <td>${h.date}</td>
        <td>${h.time || "—"}</td>
        <td><span class="badge badge-success">${h.status}</span></td>
        <td>${h.confidence ? (h.confidence * 100).toFixed(1) + "%" : "Manual"}</td>
      </tr>
    `).join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="4" style="color: var(--danger); text-align: center;">Failed to load history.</td></tr>`;
  }
}

function closeHistoryModal() {
  document.getElementById("historyModal").classList.remove("active");
}

async function deleteStudent(studentId) {
  if (!confirm(`Are you sure you want to delete student ${studentId}? All attendance history and facial biometric data will be permanently removed.`)) {
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/students/${studentId}`, { method: "DELETE" });
    if (res.ok) {
      showToast(`Student ${studentId} deleted successfully.`, "success");
      loadStudentDirectory();
      loadTodayAttendance();
    } else {
      throw new Error();
    }
  } catch {
    showToast("Failed to delete student.", "danger");
  }
}

// ==================== ENROLLMENT TAB ====================
let enrollStream = null;
async function startEnrollCamera() {
  if (enrollStream) return;
  try {
    enrollStream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
    enrollVideo.srcObject = enrollStream;
  } catch (err) {
    showToast("Enrollment camera unavailable: " + err.message, "danger");
  }
}

function captureEnrollmentSample() {
  if (!enrollStream || enrollVideo.videoWidth === 0) {
    showToast("Start enrollment camera first", "warning");
    return;
  }
  if (capturedSamples.length >= 5) {
    showToast("Maximum 5 face samples reached.", "warning");
    return;
  }

  const canvas = document.createElement("canvas");
  canvas.width = enrollVideo.videoWidth;
  canvas.height = enrollVideo.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(enrollVideo, 0, 0);

  canvas.toBlob(blob => {
    if (!blob) return;
    capturedSamples.push(blob);
    renderSampleThumbnails();
    document.getElementById("sampleCount").textContent = capturedSamples.length;
    showToast(`Captured sample #${capturedSamples.length}!`, "success");
  }, "image/jpeg", 0.95);
}

function renderSampleThumbnails() {
  const container = document.getElementById("samplePreviews");
  container.innerHTML = "";
  capturedSamples.forEach((blob, idx) => {
    const url = URL.createObjectURL(blob);
    const img = document.createElement("img");
    img.className = "sample-thumb";
    img.src = url;
    img.title = `Sample #${idx + 1}`;
    container.appendChild(img);
  });
}

function resetEnrollmentSamples() {
  capturedSamples = [];
  document.getElementById("samplePreviews").innerHTML = "";
  document.getElementById("sampleCount").textContent = "0";
}

async function handleEnrollmentSubmit(e) {
  e.preventDefault();
  if (capturedSamples.length < 3) {
    showToast("Please capture at least 3 face samples before submitting.", "danger");
    return;
  }

  const id = document.getElementById("enrollStudentId").value.trim().toUpperCase();
  const name = document.getElementById("enrollName").value.trim();
  const dept = document.getElementById("enrollDept").value.trim();
  const sem = document.getElementById("enrollSemester").value;

  const formData = new FormData();
  formData.append("student_id", id);
  formData.append("name", name);
  formData.append("department", dept);
  formData.append("semester", sem);

  capturedSamples.forEach((blob, index) => {
    formData.append("images", blob, `sample_${index + 1}.jpg`);
  });

  const btnSubmit = document.getElementById("btnSubmitEnroll");
  const spinner = document.getElementById("enrollSpinner");
  btnSubmit.disabled = true;
  spinner.style.display = "inline-flex";

  try {
    const res = await fetch(`${API_BASE}/students/register-with-faces`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) {
      const errDetail = typeof data.detail === "object" ? data.detail.message : data.detail;
      throw new Error(errDetail || "Registration failed");
    }

    showToast(`🎉 Success! Student ${id} registered with ${data.samples_used} face embeddings!`, "success");
    document.getElementById("enrollmentForm").reset();
    resetEnrollmentSamples();
    loadStudentDirectory();
  } catch (err) {
    showToast("Registration error: " + err.message, "danger");
  } finally {
    btnSubmit.disabled = false;
    spinner.style.display = "none";
  }
}

// ==================== MONTHLY ANALYTICS ====================
function initYearMonthSelectors() {
  const monthSel = document.getElementById("selectMonth");
  const yearSel = document.getElementById("selectYear");

  const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  monthSel.innerHTML = monthNames.map((m, idx) => `<option value="${idx + 1}">${m}</option>`).join("");
  monthSel.value = new Date().getMonth() + 1;

  const currentYear = new Date().getFullYear();
  yearSel.innerHTML = [currentYear - 1, currentYear, currentYear + 1]
    .map(y => `<option value="${y}">${y}</option>`).join("");
  yearSel.value = currentYear;
}

let monthlyRecords = [];
async function fetchMonthlySummary() {
  const month = document.getElementById("selectMonth").value;
  const year = document.getElementById("selectYear").value;
  const dept = document.getElementById("filterDeptMonthly").value;

  const tbody = document.getElementById("monthlyTableBody");
  tbody.innerHTML = `<tr><td colspan="8" style="text-align: center;">Calculating monthly metrics...</td></tr>`;

  try {
    let url = `${API_BASE}/attendance/monthly?year=${year}&month=${month}`;
    if (dept) url += `&department=${encodeURIComponent(dept)}`;
    const res = await fetch(url);
    monthlyRecords = await res.json();

    if (monthlyRecords.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No attendance activity logged for this month.</td></tr>`;
      return;
    }

    tbody.innerHTML = monthlyRecords.map(m => {
      const isLow = m.attendance_percentage < 75;
      return `
        <tr>
          <td><strong>${m.student_id}</strong></td>
          <td>${m.name}</td>
          <td>${m.department}</td>
          <td>Semester ${m.semester}</td>
          <td>${m.total_active_days} days</td>
          <td>${m.present_days} days</td>
          <td>
            <span class="badge ${isLow ? 'badge-danger' : 'badge-success'}">
              ${m.attendance_percentage}%
            </span>
          </td>
          <td>${isLow ? '<span style="color: var(--danger); font-weight:600;">⚠️ Low Attendance</span>' : 'Good'}</td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" style="color: var(--danger); text-align: center;">Failed to generate monthly summary.</td></tr>`;
  }
}

// ==================== CSV EXPORTERS ====================
function exportTodayCSV() {
  if (todayRecords.length === 0) {
    showToast("No records to export.", "warning");
    return;
  }
  const headers = ["Student_ID", "Name", "Department", "Semester", "Date", "Time", "Status", "Confidence"];
  const rows = todayRecords.map(r => [
    r.student_id, `"${r.student_name || ''}"`, `"${r.department || ''}"`, r.semester || '', r.date, r.time || '', r.status, r.confidence || ''
  ]);
  downloadCSV(`Attendance_${document.getElementById("filterDate").value}.csv`, [headers, ...rows]);
}

function exportMonthlyCSV() {
  if (monthlyRecords.length === 0) {
    showToast("No monthly records to export.", "warning");
    return;
  }
  const headers = ["Student_ID", "Name", "Department", "Semester", "Year", "Month", "Total_Days", "Present_Days", "Percentage"];
  const rows = monthlyRecords.map(m => [
    m.student_id, `"${m.name}"`, `"${m.department}"`, m.semester, m.year, m.month, m.total_active_days, m.present_days, m.attendance_percentage
  ]);
  downloadCSV(`Monthly_Summary_${document.getElementById("selectYear").value}_${document.getElementById("selectMonth").value}.csv`, [headers, ...rows]);
}

function downloadCSV(filename, data) {
  const csvContent = "data:text/csv;charset=utf-8," + data.map(e => e.join(",")).join("\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast(`Downloaded ${filename}`, "success");
}

function populateDepartmentFilters() {
  const depts = new Set();
  allStudentsList.forEach(s => s.department && depts.add(s.department));
  todayRecords.forEach(r => r.department && depts.add(r.department));

  const options = ['<option value="">All Departments</option>', ...Array.from(depts).map(d => `<option value="${d}">${d}</option>`)].join("");
  document.getElementById("filterDeptToday").innerHTML = options;
  document.getElementById("filterDeptDirectory").innerHTML = options;
  document.getElementById("filterDeptMonthly").innerHTML = options;
}

// ==================== TOAST HELPER ====================
function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => container.removeChild(toast), 300);
  }, 4000);
}

// ==================== EVENT LISTENERS ====================
function setupEventListeners() {
  document.getElementById("btnToggleCamera").addEventListener("click", () => {
    if (cameraStream) stopCamera();
    else startCamera();
  });
  document.getElementById("btnStartCamera").addEventListener("click", startCamera);
  document.getElementById("btnScanOnce").addEventListener("click", scanFrame);
  document.getElementById("chkAutoScan").addEventListener("change", e => {
    if (e.target.checked) startAutoScan();
    else stopAutoScan();
  });
  document.getElementById("btnClearLog").addEventListener("click", () => {
    document.getElementById("liveDetectionLog").innerHTML = `<p style="color: var(--text-muted); font-size: 0.9rem;">Log cleared.</p>`;
  });

  document.getElementById("filterDate").addEventListener("change", loadTodayAttendance);
  document.getElementById("filterDeptToday").addEventListener("change", renderTodayTable);
  document.getElementById("searchToday").addEventListener("input", renderTodayTable);
  document.getElementById("btnExportTodayCSV").addEventListener("click", exportTodayCSV);

  document.getElementById("filterDeptDirectory").addEventListener("change", renderDirectoryTable);
  document.getElementById("searchDirectory").addEventListener("input", renderDirectoryTable);

  document.getElementById("btnCaptureSample").addEventListener("click", captureEnrollmentSample);
  document.getElementById("btnResetSamples").addEventListener("click", resetEnrollmentSamples);
  document.getElementById("enrollmentForm").addEventListener("submit", handleEnrollmentSubmit);

  document.getElementById("btnFetchMonthly").addEventListener("click", fetchMonthlySummary);
  document.getElementById("btnExportMonthlyCSV").addEventListener("click", exportMonthlyCSV);
}
