// Times the reference SimplePIR implementation (github.com/ahenzinger/simplepir,
// commit e9020b0) on a random database with the same dimensions as one blind-adapters
// layout: M = query length (our rows m), L = answer length (our cells per row).
// Single thread, one query over the whole database; the hint is skipped (FakeSetup).
package main

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/ahenzinger/simplepir/pir"
)

func main() {
	l, _ := strconv.ParseUint(os.Args[1], 10, 64) // answer length
	m, _ := strconv.ParseUint(os.Args[2], 10, 64) // query length
	reps, _ := strconv.Atoi(os.Args[3])
	pi := pir.SimplePIR{}
	p := pi.PickParamsGivenDimensions(l, m, 1024, 32)
	logp := uint64(0)
	for (uint64(1) << (logp + 1)) <= p.P {
		logp++
	}
	DB := pir.MakeRandomDB(l*m, logp, &p)
	shared := pi.Init(DB.Info, p)
	server, _ := pi.FakeSetup(DB, p)
	best := 1e18
	for r := 0; r < reps; r++ {
		_, q := pi.Query(0, shared, p, DB.Info)
		var qs pir.MsgSlice
		qs.Data = append(qs.Data, q)
		start := time.Now()
		pi.Answer(DB, qs, server, shared, p)
		s := time.Since(start).Seconds()
		if s < best {
			best = s
		}
	}
	fmt.Printf("RESULT p=%d l=%d m=%d seconds=%f\n", p.P, l, m, best)
}
