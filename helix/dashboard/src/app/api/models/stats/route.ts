import { NextResponse } from 'next/server';
import { getAllModelStats } from '@/lib/model-stats';

/** GET /api/models/stats — returns per-model stats (inference count, ratings) */
export async function GET() {
  return NextResponse.json(getAllModelStats());
}
